#include "RAGManager.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct RetrievalFixture {
    const char* query;
    int topK;
    std::vector<std::string> expectedInOrder;
};

bool checkFixture(RAGManager& rag, const RetrievalFixture& fixture) {
    const std::string context =
        rag.retrieveContextRanked(fixture.query, fixture.topK, "", "");
    std::size_t previous = 0;
    bool first = true;

    for (const std::string& expected : fixture.expectedInOrder) {
        const std::size_t position = context.find(expected);
        if (position == std::string::npos) {
            std::cerr << "FAIL query '" << fixture.query
                      << "': missing expected result '" << expected << "'\n";
            return false;
        }
        if (!first && position <= previous) {
            std::cerr << "FAIL query '" << fixture.query
                      << "': expected ranking order was not preserved\n";
            return false;
        }
        previous = position;
        first = false;
    }
    return true;
}

} // namespace

int main() {
    auto& rag = RAGManager::getInstance();
    rag.clearForTesting();

    rag.addDocument("parse json request body and validate required fields", "src/http/json_request.cpp", "cpp");
    rag.addDocument("serialize json response body with status code", "src/http/json_response.cpp", "cpp");
    rag.addDocument("validate bearer authentication token before request dispatch", "src/security/auth.cpp", "cpp");
    rag.addDocument("refresh oauth access token when credentials expire", "src/security/oauth.cpp", "cpp");
    rag.addDocument("execute sql transaction and rollback database changes on failure", "src/db/transaction.cpp", "cpp");
    rag.addDocument("prepare sql select query using bound parameters", "src/db/query.cpp", "cpp");
    rag.addDocument("render directx texture shader and vertex buffer", "src/render/directx.cpp", "cpp");
    rag.addDocument("python training loop computes optimizer loss and gradient", "train/model.py", "python");

    const std::vector<RetrievalFixture> fixtures = {
        {"json request validation", 2,
         {"src/http/json_request.cpp", "src/http/json_response.cpp"}},
        {"bearer authentication token", 2,
         {"src/security/auth.cpp", "src/security/oauth.cpp"}},
        {"sql transaction rollback", 2,
         {"src/db/transaction.cpp", "src/db/query.cpp"}},
        {"directx texture shader", 1,
         {"src/render/directx.cpp"}},
        {"python optimizer gradient", 1,
         {"train/model.py"}},
    };

    // Structured source results must be ranked, unique, and safe for evaluation.
    bool ok = true;
    const auto rankedPaths = rag.retrieveRankedSources("json request validation", 3);
    if (rankedPaths.empty() || rankedPaths.front() != "src/http/json_request.cpp" ||
        rankedPaths.size() > 3) {
        std::cerr << "Structured RAG paths: unexpected ranking or result count.\\n";
        ok = false;
    }
    const auto noHits = rag.retrieveRankedSources("zzzz_unknown_never_seen_913", 3);
    if (!noHits.empty() || !rag.retrieveRankedSources("", 3).empty() ||
        !rag.retrieveRankedSources("json request", 0).empty()) {
        std::cerr << "Structured RAG paths: zero-hit or invalid-query behavior.\\n";
        ok = false;
    }
    rag.addDocument("json request validation additional unique chunk", "src/http/json_request.cpp", "cpp");
    const auto uniquePaths = rag.retrieveRankedSources("json request validation", 5);
    if (std::count(uniquePaths.begin(), uniquePaths.end(), "src/http/json_request.cpp") != 1) {
        std::cerr << "Structured RAG paths: duplicate source.\\n";
        ok = false;
    }

    for (const RetrievalFixture& fixture : fixtures)
        ok &= checkFixture(rag, fixture);

    // Replacing one source must preserve unrelated keys and avoid stale chunks.
    rag.updateSource("old sentinel_alpha payload", "src/updated.cpp", "cpp");
    rag.updateSource("unrelated sentinel_beta payload", "src/untouched.cpp", "cpp");
    rag.updateSource("new sentinel_gamma payload", "src/updated.cpp", "cpp");
    const std::string newContext = rag.retrieveContextRanked("sentinel_gamma", 3, "", "");
    const std::string oldContext = rag.retrieveContextRanked("sentinel_alpha", 3, "", "");
    const std::string unrelatedContext = rag.retrieveContextRanked("sentinel_beta", 3, "", "");
    if (newContext.find("src/updated.cpp") == std::string::npos ||
        oldContext.find("old sentinel_alpha payload") != std::string::npos ||
        unrelatedContext.find("src/untouched.cpp") == std::string::npos) {
        std::cerr << "Incremental source replacement lost or retained invalid documents.\\n";
        ok = false;
    }
    // Re-adding the same source after replacement must still deduplicate.
    rag.addDocument("new sentinel_gamma payload", "src/updated.cpp", "cpp");
    rag.updateSource("", "src/updated.cpp", "cpp");
    if (rag.retrieveContextRanked("sentinel_gamma", 3, "src/updated.cpp", "").find("src/updated.cpp") != std::string::npos) {
        std::cerr << "Empty source update did not remove indexed documents.\\n";
        ok = false;
    }

    rag.clearForTesting();
    if (!ok)
        return 1;

    std::cout << "RAG retrieval-quality fixtures passed: "
              << fixtures.size() << " representative queries.\n";
    return 0;
}
