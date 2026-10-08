#include "RAGManager.h"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

struct Fixture {
    const char* query;
    const char* relevantSource;
};

int main() {
    auto& rag = RAGManager::getInstance();
#ifdef NPPAI_TESTING
    rag.clearForTesting();
#endif
    rag.addDocument("parse json request body and validate required fields", "src/http/json_request.cpp", "cpp");
    rag.addDocument("serialize json response body with status code", "src/http/json_response.cpp", "cpp");
    rag.addDocument("validate bearer authentication token before request dispatch", "src/security/auth.cpp", "cpp");
    rag.addDocument("refresh oauth access token when credentials expire", "src/security/oauth.cpp", "cpp");
    rag.addDocument("execute sql transaction and rollback database changes on failure", "src/db/transaction.cpp", "cpp");
    rag.addDocument("prepare sql select query using bound parameters", "src/db/query.cpp", "cpp");
    rag.addDocument("render directx texture shader and vertex buffer", "src/render/directx.cpp", "cpp");
    rag.addDocument("python training loop computes optimizer loss and gradient", "train/model.py", "python");

    // Hard negatives share vocabulary with the relevant document but address different tasks.
    rag.addDocument("validate json response body and required fields for response schema", "src/http/response_schema.cpp", "cpp");
    rag.addDocument("parse json request body for debug tracing and metrics", "src/http/request_trace.cpp", "cpp");
    rag.addDocument("authentication token refresh endpoint logs expired bearer credentials", "src/security/token_logs.cpp", "cpp");
    rag.addDocument("sql transaction report of rollback attempts and audit changes", "src/db/audit.cpp", "cpp");
    rag.addDocument("render directx texture shader cache and debugging overlays", "src/render/debug.cpp", "cpp");
    rag.addDocument("python optimizer loss gradient logging and metric dashboard", "train/metrics.py", "python");
    rag.addDocument("oauth access token refresh instructions for documentation", "docs/oauth.md", "md");
    rag.addDocument("sql bound parameters in tutorial sample", "docs/sql.md", "md");

    const std::vector<Fixture> fixtures = {
        {"json request validation", "src/http/json_request.cpp"},
        {"bearer authentication token", "src/security/auth.cpp"},
        {"sql transaction rollback", "src/db/transaction.cpp"},
        {"directx texture shader", "src/render/directx.cpp"},
        {"python optimizer gradient", "train/model.py"},
        {"oauth credentials expire", "src/security/oauth.cpp"},
        {"sql bound parameters", "src/db/query.cpp"},
        {"json response status code", "src/http/json_response.cpp"},
        {"json request required fields", "src/http/json_request.cpp"},
        {"validate json response schema", "src/http/response_schema.cpp"},
        {"json request debug tracing", "src/http/request_trace.cpp"},
        {"expired bearer credentials logging", "src/security/token_logs.cpp"},
        {"sql rollback audit changes", "src/db/audit.cpp"},
        {"directx shader cache debugging", "src/render/debug.cpp"},
        {"python gradient metrics dashboard", "train/metrics.py"},
        {"oauth refresh documentation", "docs/oauth.md"},
        {"sql parameters tutorial", "docs/sql.md"}
    };
    struct Weights { float cosine; float lexical; };
    const std::vector<Weights> variants = {{0.35f,0.65f},{0.50f,0.50f},{0.65f,0.35f},{0.80f,0.20f}};
    for (const auto& w : variants) {
        int hits = 0;
        double reciprocalRankSum = 0.0;
        for (const auto& fixture : fixtures) {
            const std::string result = rag.retrieveContextRankedWeighted(fixture.query, 3, "", "", w.cosine, w.lexical);
            const auto relevantPos = result.find(fixture.relevantSource);
            if (relevantPos == std::string::npos) continue;
            int rank = 1;
            size_t pos = 0;
            while ((pos = result.find("--- Zapisany Kontekst RAG", pos)) != std::string::npos && pos < relevantPos) {
                ++rank;
                ++pos;
            }
            --rank;
            if (rank >= 1 && rank <= 3) {
                ++hits;
                reciprocalRankSum += 1.0 / rank;
            }
        }
        std::cout << "{\"cosine_weight\":" << w.cosine
                  << ",\"lexical_weight\":" << w.lexical
                  << ",\"queries\":" << fixtures.size()
                  << ",\"recall_at_3\":" << static_cast<double>(hits) / fixtures.size()
                  << ",\"mrr_at_3\":" << reciprocalRankSum / fixtures.size()
                  << "}" << std::endl;
    }
    return 0;
}
