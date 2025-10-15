#pragma once
#include <string>
#include <vector>
#include <optional>

// Simple rule: "*" = all talkgroups
// An item like "1200-1299" is a range, otherwise parse as integer.
struct TalkgroupRule {
    bool all{false};
    std::vector<int> singles;          // e.g., [1201, 34320]
    std::vector<std::pair<int,int>> ranges; // e.g., {{1000,1999}}

    bool matches(int tg) const {
        if (all) return true;
        for (int v : singles) if (tg == v) return true;
        for (auto& r : ranges) if (tg >= r.first && tg <= r.second) return true;
        return false;
    }
};

struct TranscribeApi {
    bool        enabled{false};
    std::string endpoint;          // e.g. https://api.openai.com/v1/audio/transcriptions
    std::string model;             // e.g. whisper-1
    std::string api_key;           // Bearer token (or put empty to rely on env var)
    long        connect_timeout_ms{10000};
    long        transfer_timeout_ms{0}; // 0 = no limit
    bool        verify_tls{true};
};

struct TranscribeSystemConfig {
    std::string short_name;
    bool        enabled{true};
    std::string audio{"auto"};     // "auto" prefer m4a else wav
    TalkgroupRule tg_rule;

    TranscribeApi api;
};

