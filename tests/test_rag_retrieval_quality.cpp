#include "RAGManager.h"
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
    const std::string context = rag.retrieveContextRanked(fixture.query, fixture.topK);
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

    bool ok = true;
    for (const RetrievalFixture& fixture : fixtures)
        ok &= checkFixture(rag, fixture);

    rag.clearForTesting();
    if (!ok)
        return 1;

    std::cout << "RAG retrieval-quality fixtures passed: "
              << fixtures.size() << " representative queries.\n";
    return 0;
}
