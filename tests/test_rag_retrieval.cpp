#include "RAGManager.h"
#include <iostream>
#include <string>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}
}

int main() {
    bool ok = true;
    auto& rag = RAGManager::getInstance();
    rag.clearForTesting();

    rag.addDocument("parser validates json request payload");
    rag.addDocument("renderer draws triangle texture shader");
    rag.addDocument("json parser rejects invalid request");

    const std::string ranked = rag.retrieveContext("json parser request", 2);
    const auto first = ranked.find("json parser rejects invalid request");
    const auto second = ranked.find("parser validates json request payload");
    ok &= check(first != std::string::npos && second != std::string::npos && first < second,
                "more relevant document ranks first");
    ok &= check(ranked.find("renderer draws triangle") == std::string::npos,
                "topK excludes unrelated lower-ranked documents");

    rag.clearForTesting();
    rag.addDocument("beta shared token");
    rag.addDocument("alpha shared token");
    const std::string tied = rag.retrieveContext("shared token", 2);
    ok &= check(tied.find("alpha shared token") < tied.find("beta shared token"),
                "equal-score results use deterministic text ordering");

    rag.clearForTesting();
    rag.addDocument("duplicate retrieval document");
    rag.addDocument("duplicate retrieval document");
    const std::string deduped = rag.retrieveContext("duplicate retrieval document", 3);
    const auto pos = deduped.find("duplicate retrieval document");
    ok &= check(pos != std::string::npos &&
                    deduped.find("duplicate retrieval document", pos + 1) == std::string::npos,
                "identical documents are indexed only once");

    rag.clearForTesting();
    rag.addDocument("parser parser parser unrelated");
    rag.addDocument("parser authentication bearer token");
    const std::string lexical = rag.retrieveContext("parser authentication bearer token", 1);
    ok &= check(lexical.find("parser authentication bearer token") != std::string::npos,
                "exact query-token coverage improves ranking beyond term frequency");

    rag.clearForTesting();
    const std::string filler(1100, 'x');
    const std::string longDocument =
        filler + "\n"
        "authentication middleware validates bearer token before request handling\n" +
        filler;
    rag.addDocument(longDocument);
    const std::string chunked = rag.retrieveContext("authentication bearer token", 1);
    ok &= check(chunked.find("authentication middleware validates bearer token") != std::string::npos,
                "long documents expose the relevant chunk to retrieval");
    ok &= check(chunked.find(longDocument) == std::string::npos,
                "long documents are indexed as bounded chunks instead of one monolith");

    rag.clearForTesting();
    if (!ok) return 1;
    std::cout << "RAG retrieval tests passed.\n";
    return 0;
}