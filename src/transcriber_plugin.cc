#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <memory>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cctype>

#include <boost/log/trivial.hpp>
#include <boost/dll/alias.hpp>
#include <json.hpp>
#include <curl/curl.h>

// TR plugin API (no dependency on other plugins)
#include <trunk-recorder/plugin_manager/plugin_api.h>

#include "transcriber_config.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

static constexpr const char* TAG = "\t[Transcriber]\t";
#define TLOG(sev) BOOST_LOG_TRIVIAL(sev) << TAG

// ----------------- small local helpers (no external util.hpp) -----------------
static bool file_exists(const std::string& p) {
    std::error_code ec;
    return !p.empty() && fs::exists(p, ec);
}

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

static std::string basename_of(const std::string& p) {
    try { return fs::path(p).filename().string(); }
    catch (...) { return p; }
}

static size_t append_body(char* ptr, size_t size, size_t nmemb, void* userdata) {
    const size_t total = size * nmemb;
    static_cast<std::string*>(userdata)->append(ptr, total);
    return total;
}

static bool load_json_with_retries(const std::string& path,
                                   nlohmann::json* out,
                                   int tries = 10,
                                   int delay_ms = 50) {
    for (int i = 0; i < tries; ++i) {
        try {
            std::ifstream in(path);
            if (!in.good()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                continue;
            }
            *out = nlohmann::json::parse(in, /*cb*/nullptr, /*allow_exceptions*/true, /*ignore_comments*/true);
            return true;
        } catch (...) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
    }
    return false;
}

// Atomic JSON writer: write to .tmp then rename over target
static bool atomic_write_json_file(const std::string& path, const nlohmann::json& j) {
    try {
        fs::path p(path);
        fs::path tmp = p; tmp += ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            out << j.dump(2);
            out.flush();
            if (!out) return false;
        }
        std::error_code ec;
        fs::rename(tmp, p, ec);                 // atomic on POSIX
        if (ec) {
            fs::remove(p, ec);                    // best-effort remove then retry
            ec.clear();
            fs::rename(tmp, p, ec);
            if (ec) return false;
        }
        return true;
    } catch (...) { return false; }
}


static void set_transcriber_status(const std::string& json_path,
                                   const std::string& status,
                                   const std::string& model = {},
                                   std::optional<int> talkgroup = std::nullopt,
                                   std::optional<std::string> error = std::nullopt)
{
    if (json_path.empty() || !file_exists(json_path)) return;

    try {
        nlohmann::json j;
        if (!load_json_with_retries(json_path, &j)) {
            // best-effort: start a fresh object if concurrent writer mid-write
            j = nlohmann::json::object();
        }

        nlohmann::json& node = j["transcriber"];
        if (!node.is_object()) node = nlohmann::json::object();

        node["status"] = status;
        if (!model.empty())       node["model"] = model;
        if (talkgroup.has_value()) node["talkgroup"] = *talkgroup;
        if (error.has_value())     node["error"]     = *error;
        else if (node.contains("error")) node.erase("error");
        node["updated_at"] = static_cast<long>(std::time(nullptr));

        (void) atomic_write_json_file(json_path, j);
    } catch (...) {
        // best-effort; ignore errors
    }
}


static void write_transcript_done(const std::string& json_path,
                                  const std::string& text,
                                  const std::string& model,
                                  int talkgroup)
{
    if (json_path.empty() || !file_exists(json_path)) return;

    try {
        nlohmann::json j;
        if (!load_json_with_retries(json_path, &j)) {
            // if we can't read, still produce a minimal valid JSON
            j = nlohmann::json::object();
        }

        nlohmann::json& node = j["transcriber"];
        if (!node.is_object()) node = nlohmann::json::object();

        node["status"]     = "done";
        node["transcript"] = text;
        node["model"]      = model;
        node["talkgroup"]  = talkgroup;
        node["updated_at"] = static_cast<long>(std::time(nullptr));

        if (!atomic_write_json_file(json_path, j)) {
            TLOG(warning) << "Atomic write failed for transcript JSON: " << json_path;
        }
    } catch (const std::exception& e) {
        TLOG(warning) << "Failed updating JSON transcript: " << e.what();
    }
}



// --------------- talkgroup rule parser ---------------
static TalkgroupRule parse_tg_rule(const json& jtg) {
    TalkgroupRule rule;
    rule.all = false; // if provided, be explicit; otherwise default set later

    auto push_range = [&](int a, int b){
        if (a > b) std::swap(a,b);
        rule.ranges.push_back({a,b});
    };

    auto parse_one = [&](const std::string& s){
        if (s == "*") { rule.all = true; rule.singles.clear(); rule.ranges.clear(); return; }
        auto dash = s.find('-');
        if (dash != std::string::npos) {
            int a = std::stoi(s.substr(0, dash));
            int b = std::stoi(s.substr(dash+1));
            push_range(a,b);
        } else {
            rule.singles.push_back(std::stoi(s));
        }
    };

    if (jtg.is_string()) {
        parse_one(jtg.get<std::string>());
    } else if (jtg.is_array()) {
        for (const auto& el : jtg) {
            if (el.is_string()) parse_one(el.get<std::string>());
            else if (el.is_number_integer()) rule.singles.push_back(el.get<int>());
        }
    } else {
        // if malformed, treat as all
        rule.all = true;
        rule.singles.clear();
        rule.ranges.clear();
    }
    return rule;
}

// --------------- worker job ---------------
struct TranscribeJob {
    std::string audio_wav;
    std::string audio_m4a;
    std::string json_path;
    std::string short_name;
    std::time_t start_time{0};
    int         talkgroup{0};
};

// --------------- per-system ctx ---------------
struct SystemCtx {
    TranscribeSystemConfig cfg;
    explicit SystemCtx(const TranscribeSystemConfig& c) : cfg(c) {}
};

// --------------- worker ---------------
class TranscriberWorker {
public:
    explicit TranscriberWorker(const std::unordered_map<std::string, TranscribeSystemConfig>& by_sys) {
        for (const auto& kv : by_sys) {
            systems_.emplace(kv.first, std::make_unique<SystemCtx>(kv.second));
        }
    }

    void start() { stop_ = false; worker_ = std::thread([this]{ run(); }); }
    void stop()  { { std::lock_guard<std::mutex> lk(mu_); stop_ = true; } cv_.notify_all(); if (worker_.joinable()) worker_.join(); }
    void enqueue(const TranscribeJob& j) { { std::lock_guard<std::mutex> lk(mu_); q_.push(j); } cv_.notify_one(); }

private:
    static bool choose_audio(const TranscribeSystemConfig& cfg,
                             const TranscribeJob& job,
                             std::string& chosen)
    {
        const bool has_wav = file_exists(job.audio_wav);
        const bool has_m4a = file_exists(job.audio_m4a);
        const auto mode = to_lower_copy(cfg.audio);

        if (mode == "m4a") {
            if (has_m4a) chosen = job.audio_m4a;
            else if (has_wav) chosen = job.audio_wav;
        } else if (mode == "wav") {
            if (has_wav) chosen = job.audio_wav;
            else if (has_m4a) chosen = job.audio_m4a;
        } else { // auto
            if (has_m4a) chosen = job.audio_m4a;
            else if (has_wav) chosen = job.audio_wav;
        }
        return !chosen.empty();
    }

    static bool post_transcribe(const TranscribeSystemConfig& cfg,
                                const std::string& audio_path,
                                std::string* transcript_out,
                                std::string* err_out)
    {
        if (!cfg.api.enabled) return true; // treat as no-op success

        CURL* curl = curl_easy_init();
        if (!curl) { if (err_out) *err_out = "curl_easy_init failed"; return false; }

        std::string resp;
        struct curl_slist* headers = nullptr;

        // Bearer auth
        std::string key = cfg.api.api_key;
        if (key.empty()) {
            if (const char* envk = std::getenv("OPENAI_API_KEY")) key = envk;
        }
        if (!key.empty()) {
            std::string h = "Authorization: Bearer " + key;
            headers = curl_slist_append(headers, h.c_str());
        }

        // multipart/form-data
        curl_mime* mime = curl_mime_init(curl);
        curl_mimepart* part = nullptr;

        // file (many endpoints expect field name "file" or "audio")
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "file");
        curl_mime_filedata(part, audio_path.c_str());

        // model
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "model");
        curl_mime_data(part, cfg.api.model.c_str(), CURL_ZERO_TERMINATED);

        // response format (for OpenAI whisper endpoints)
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "response_format");
        curl_mime_data(part, "verbose_json", CURL_ZERO_TERMINATED);

        curl_easy_setopt(curl, CURLOPT_URL, cfg.api.endpoint.c_str());
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, cfg.api.connect_timeout_ms);
        if (cfg.api.transfer_timeout_ms > 0) curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, cfg.api.transfer_timeout_ms);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, append_body);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, cfg.api.verify_tls ? 1L : 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, cfg.api.verify_tls ? 2L : 0L);

        if (const char* v = std::getenv("TR_TRANSCRIBE_VERBOSE")) {
            if (std::string(v) == "1" || std::string(v) == "true") curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
        }

        TLOG(info) << "POST " << cfg.api.endpoint << " model=" << cfg.api.model
                   << " file=" << basename_of(audio_path);

        long http_code = 0;
        CURLcode rc = curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_mime_free(mime);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (rc != CURLE_OK || (http_code/100) != 2) {
            if (err_out) *err_out = "HTTP " + std::to_string(http_code) +
                                    " CURL " + std::string(curl_easy_strerror(rc)) +
                                    " body: " + (resp.size() > 512 ? resp.substr(0,512)+"..." : resp);
            return false;
        }

        try {
            auto j = json::parse(resp);
            if (j.contains("text") && j["text"].is_string()) {
                *transcript_out = j["text"].get<std::string>();
            } else {
                *transcript_out = j.dump(); // fallback (vendor variations)
            }
            return true;
        } catch (const std::exception& e) {
            if (err_out) *err_out = std::string("JSON parse error: ") + e.what();
            return false;
        }
    }

    void run() {
        while (true) {
            TranscribeJob job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]{ return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) break;
                job = q_.front();
                q_.pop();
            }

            // find system cfg
            auto it = systems_.find(job.short_name);
            if (it == systems_.end()) {
                TLOG(warning) << "[" << job.short_name << "] no transcriber config — skipping";
                continue;
            }
            const auto& cfg = it->second->cfg;
            if (!cfg.enabled) continue;

            // talkgroup filter
            if (!cfg.tg_rule.matches(job.talkgroup)) {
                TLOG(info) << "[" << job.short_name << "] TG " << job.talkgroup << " not selected — skip";
                continue;
            }

            // choose audio file
            std::string audio_path;
            if (!choose_audio(cfg, job, audio_path)) {
                TLOG(warning) << "[" << job.short_name << "] no audio file present — skipping";
                continue;
            }

            set_transcriber_status(job.json_path, "started", cfg.api.model, job.talkgroup);

            std::string transcript, err;
            if (!post_transcribe(cfg, audio_path, &transcript, &err)) {
                TLOG(error) << "[" << job.short_name << "] transcribe failed: " << err;
                set_transcriber_status(job.json_path, "error", cfg.api.model, job.talkgroup, err);
                continue;
            }

            write_transcript_done(job.json_path, transcript, cfg.api.model, job.talkgroup);
            TLOG(info) << "[" << job.short_name << "] transcript written (" << basename_of(job.json_path) << ")";

        }
    }

private:
    std::unordered_map<std::string, std::unique_ptr<SystemCtx>> systems_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<TranscribeJob> q_;
    std::thread worker_;
    bool stop_{false};
};

// =============================
//   Plugin class
// =============================
class Transcriber_Plugin : public Plugin_Api {
public:
    Transcriber_Plugin() = default;

    int parse_config(json cfg_root) override {
        try {
            TLOG(info) << "Parsing transcriber config...";
            cfg_by_system_.clear();

            if (!cfg_root.contains("systems") || !cfg_root["systems"].is_array()) {
                TLOG(error) << "config missing \"systems\" array";
                return -1;
            }

            for (const auto& el : cfg_root["systems"]) {
                if (!el.contains("shortName")) continue;

                TranscribeSystemConfig sc;
                sc.short_name = el.value("shortName", "");
                if (sc.short_name.empty()) continue;

                sc.enabled = el.value("enabled", true);
                sc.audio   = to_lower_copy(el.value("audio", "auto"));

                if (el.contains("talkgroups")) {
                    sc.tg_rule = parse_tg_rule(el["talkgroups"]);
                } else {
                    sc.tg_rule.all = true; // default to all if not specified
                }

                if (el.contains("api")) {
                    const auto& a = el["api"];
                    sc.api.enabled             = a.value("enabled", true);
                    sc.api.endpoint            = a.value("endpoint", sc.api.endpoint);
                    sc.api.model               = a.value("model", sc.api.model);
                    sc.api.api_key             = a.value("api_key", "");
                    sc.api.connect_timeout_ms  = a.value("connect_timeout_ms", sc.api.connect_timeout_ms);
                    sc.api.transfer_timeout_ms = a.value("transfer_timeout_ms", sc.api.transfer_timeout_ms);
                    sc.api.verify_tls          = a.value("verify_tls", sc.api.verify_tls);
                }

                cfg_by_system_.emplace(sc.short_name, std::move(sc));
            }

            if (cfg_by_system_.empty()) {
                TLOG(error) << "no valid systems configured";
                return -1;
            }
        } catch (const std::exception& e) {
            TLOG(error) << "parse_config error: " << e.what();
            return -1;
        }

        TLOG(info) << "Loaded " << cfg_by_system_.size() << " system config"
                   << (cfg_by_system_.size() == 1 ? "" : "s") << ".";
        return 0;
    }

    int init(Config*, std::vector<Source*>, std::vector<System*>) override {
        TLOG(info) << "Initializing transcriber (curl + worker)...";
        curl_global_init(CURL_GLOBAL_DEFAULT);
        worker_ = std::make_unique<TranscriberWorker>(cfg_by_system_);
        worker_->start();
        return 0;
    }

    int start() override { TLOG(info) << "Worker started."; return 0; }

    int stop() override {
        TLOG(info) << "Stopping worker...";
        if (worker_) { worker_->stop(); worker_.reset(); }
        curl_global_cleanup();
        TLOG(info) << "Worker stopped.";
        return 0;
    }

    int call_end(Call_Data_t call_info) override {
        try {
            TranscribeJob j;
            j.audio_wav  = call_info.filename;
            j.audio_m4a  = call_info.converted;
            j.json_path  = call_info.status_filename;
            j.short_name = call_info.short_name;
            j.start_time = static_cast<std::time_t>(call_info.start_time);
            j.talkgroup  = call_info.talkgroup;

            // Derive JSON if missing (rare, but keeps things robust)
            if (j.json_path.empty()) {
                try {
                    std::string p = !j.audio_m4a.empty() ? j.audio_m4a : j.audio_wav;
                    if (!p.empty()) {
                        // simple extension swap
                        auto pos = p.find_last_of('.');
                        if (pos != std::string::npos) j.json_path = p.substr(0, pos) + ".json";
                        else                          j.json_path = p + ".json";
                    }
                } catch (...) {}
            }

            // Look up config early so we can decide whether to stamp "queued"
            auto it = cfg_by_system_.find(j.short_name);
            if (it != cfg_by_system_.end()) {
                const auto& sc = it->second;
                if (sc.enabled && sc.tg_rule.matches(j.talkgroup)) {
                    // Make MQTT "presence probe" see us immediately
                    set_transcriber_status(j.json_path, "queued", sc.api.model, j.talkgroup);
                }
            }

            if (worker_) worker_->enqueue(j);
        } catch (...) {
            return -1;
        }
        return 0;
    }

    static boost::shared_ptr<Transcriber_Plugin> create() {
        return boost::shared_ptr<Transcriber_Plugin>(new Transcriber_Plugin());
    }

private:
    std::unordered_map<std::string, TranscribeSystemConfig> cfg_by_system_;
    std::unique_ptr<TranscriberWorker> worker_;
};

BOOST_DLL_ALIAS(Transcriber_Plugin::create, create_plugin)
