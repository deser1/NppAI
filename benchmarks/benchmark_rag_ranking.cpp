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

    const std::vector<Fixture> fixtures = {
        {"json request validation", "src/http/json_request.cpp"},
        {"bearer authentication token", "src/security/auth.cpp"},
        {"sql transaction rollback", "src/db/transaction.cpp"},
        {"directx texture shader", "src/render/directx.cpp"},
        {"python optimizer gradient", "train/model.py"},
        {"oauth credentials expire", "src/security/oauth.cpp"},
        {"sql bound parameters", "src/db/query.cpp"},
        {"json response status code", "src/http/json_response.cpp"}
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
