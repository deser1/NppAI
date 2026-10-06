#include "RAGManager.h"
#include <iostream>
#include <string>
#include <cstdio>
#include <fstream>
#include <cstdint>
#include <filesystem>

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

    const std::string zeroTopK = rag.retrieveContext("json parser request", 0);
    const std::string negativeTopK = rag.retrieveContext("json parser request", -1);
    ok &= check(zeroTopK.empty() && negativeTopK.empty(),
                "non-positive topK returns no retrieval results");

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
    rag.addDocument("authentication bearer token exact coverage");
    rag.addDocument("authentication authentication authentication bearer noise noise noise");
    const std::string hybridRanked = rag.retrieveContext("authentication bearer token", 1);
    ok &= check(hybridRanked.find("exact coverage") != std::string::npos,
                "full query-token coverage receives a hybrid ranking bonus");

    rag.clearForTesting();
    rag.addDocument("validate request token", "src/api.cpp", "cpp");
    rag.addDocument("validate request token", "scripts/api.py", "python");
    const std::string cppOnly = rag.retrieveContext("validate request token", 3, "src/api.cpp", "cpp");
    ok &= check(cppOnly.find("validate request token") != std::string::npos,
                "source and language filters retain matching documents");
    ok &= check(cppOnly.find("src/api.cpp") != std::string::npos &&
                    cppOnly.find("cpp") != std::string::npos,
                "retrieved context exposes source and language metadata");
    rag.addDocument("validate request token", "scripts/api.py", "python");
    const std::string pythonOnly = rag.retrieveContext(
        "validate request token", 3, "scripts/api.py", "python");
    ok &= check(pythonOnly.find("validate request token") != std::string::npos,
                "identical text with distinct metadata remains independently retrievable");
    const std::string wrongLanguage = rag.retrieveContext("validate request token", 3, "", "rust");
    ok &= check(wrongLanguage.empty(),
                "language filter excludes non-matching documents");

    rag.clearForTesting();
    rag.addDocument("parser request validation helper", "src/shared.cpp", "cpp");
    rag.addDocument("parser request validation helper", "src/active.cpp", "cpp");
    rag.addDocument("parser request validation helper", "scripts/active.py", "python");
    const std::string rankedContext = rag.retrieveContextRanked(
        "parser request validation helper", 3, "src/active.cpp", "cpp");
    const auto activePos = rankedContext.find("Źródło: src/active.cpp");
    const auto sharedPos = rankedContext.find("Źródło: src/shared.cpp");
    const auto pythonPos = rankedContext.find("Źródło: scripts/active.py");
    ok &= check(activePos != std::string::npos && sharedPos != std::string::npos &&
                    pythonPos != std::string::npos && activePos < sharedPos && sharedPos < pythonPos,
                "context-aware ranking prefers active source then matching language without filtering alternatives");

    rag.clearForTesting();
    rag.updateSource("legacy parser implementation", "src/parser.cpp", "cpp");
    rag.updateSource("modern parser implementation", "src/parser.cpp", "cpp");
    const std::string updatedSource = rag.retrieveContext("parser implementation", 3, "src/parser.cpp", "cpp");
    ok &= check(updatedSource.find("modern parser implementation") != std::string::npos &&
                    updatedSource.find("legacy parser implementation") == std::string::npos,
                "incremental source update replaces stale indexed content");
    rag.updateSource("", "src/parser.cpp", "cpp");
    ok &= check(rag.retrieveContext("parser implementation", 3, "src/parser.cpp", "cpp").empty(),
                "empty source update removes deleted file content");

    rag.clearForTesting();
    const std::filesystem::path repoFixture = "rag_repo_fixture";
    std::filesystem::remove_all(repoFixture);
    std::filesystem::create_directories(repoFixture / "src");
    std::filesystem::create_directories(repoFixture / "node_modules");
    std::filesystem::create_directories(repoFixture / "dist");
    std::filesystem::create_directories(repoFixture / "vendor");
    std::filesystem::create_directories(repoFixture / "target");
    std::filesystem::create_directories(repoFixture / ".venv");
    { std::ofstream(repoFixture / "src" / "auth.cpp") << "repository index bearer authentication token"; }
    { std::ofstream(repoFixture / "README.md") << "repository documentation marker"; }
    { std::ofstream(repoFixture / "node_modules" / "ignored.js") << "ignored dependency marker"; }
    { std::ofstream(repoFixture / "dist" / "bundle.js") << "generated distribution marker"; }
    { std::ofstream(repoFixture / "vendor" / "library.php") << "vendored dependency marker"; }
    { std::ofstream(repoFixture / "target" / "generated.rs") << "generated rust target marker"; }
    { std::ofstream(repoFixture / ".venv" / "package.py") << "virtual environment marker"; }
    const size_t indexedFiles = rag.indexRepository(repoFixture.string());
    ok &= check(indexedFiles == 2, "repository indexer scans supported files and ignores dependency directories");
    const std::string repositoryContext = rag.retrieveContext("bearer authentication token", 1, "src/auth.cpp", "cpp");
    ok &= check(repositoryContext.find("repository index bearer authentication token") != std::string::npos,
                "repository indexer stores relative source and language metadata");
    // Verify ignored directories through isolated repository fixtures. This tests the
    // indexer's observable file count directly and avoids retrieval-score behavior.
    const auto checkIgnoredDirectory = [&](const std::string& directory,
                                           const std::string& filename,
                                           const char* message) {
        rag.clearForTesting();
        const std::filesystem::path ignoredFixture = "rag_ignored_fixture";
        std::filesystem::remove_all(ignoredFixture);
        std::filesystem::create_directories(ignoredFixture / directory);
        { std::ofstream(ignoredFixture / directory / filename) << "must not be indexed"; }
        const size_t count = rag.indexRepository(ignoredFixture.string());
        std::filesystem::remove_all(ignoredFixture);
        ok &= check(count == 0, message);
    };
    checkIgnoredDirectory("node_modules", "ignored.js",
                          "repository indexer excludes ignored directories");
    checkIgnoredDirectory("dist", "bundle.js",
                          "repository indexer excludes distribution output");
    checkIgnoredDirectory("vendor", "library.php",
                          "repository indexer excludes vendored dependencies");
    checkIgnoredDirectory("target", "generated.rs",
                          "repository indexer excludes generated target directories");
    checkIgnoredDirectory(".venv", "package.py",
                          "repository indexer excludes virtual environments");
    std::filesystem::remove_all(repoFixture);

    rag.clearForTesting();
    rag.addDocument("void parse_http_response();");
    rag.addDocument("void parseCacheEntry();");
    const std::string snakeCase = rag.retrieveContext("http response", 1);
    ok &= check(snakeCase.find("parse_http_response") != std::string::npos,
                "snake_case identifier parts contribute to lexical ranking");

    rag.clearForTesting();
    rag.addDocument("void validateBearerToken();");
    rag.addDocument("void validateCacheEntry();");
    const std::string camelCase = rag.retrieveContext("bearer token", 1);
    ok &= check(camelCase.find("validateBearerToken") != std::string::npos,
                "camelCase identifier parts contribute to lexical ranking");

    rag.clearForTesting();
    rag.addDocument("void parseHTTPResponse();");
    rag.addDocument("void parseCacheEntry();");
    const std::string acronymCamelCase = rag.retrieveContext("http response", 1);
    ok &= check(acronymCamelCase.find("parseHTTPResponse") != std::string::npos,
                "acronym boundaries in camelCase identifiers contribute to lexical ranking");

    rag.clearForTesting();
    rag.addDocument("obsługa żądania użytkownika");
    rag.addDocument("renderer texture shader");
    const std::string utf8Tokens = rag.retrieveContext("żądania użytkownika", 1);
    ok &= check(utf8Tokens.find("obsługa żądania użytkownika") != std::string::npos,
                "UTF-8 words remain intact for lexical and embedding retrieval");

    rag.clearForTesting();
    const std::string dbPath = "rag_metadata_v2_test.bin";
    rag.addDocument("persistent metadata token", "src/persist.cpp", "cpp");
    rag.saveDatabase(dbPath);
    rag.clearForTesting();
    rag.loadDatabase(dbPath);
    const std::string persisted = rag.retrieveContext(
        "persistent metadata token", 1, "src/persist.cpp", "cpp");
    ok &= check(persisted.find("persistent metadata token") != std::string::npos,
                "v2 database persists source and language metadata");
    std::remove(dbPath.c_str());

    rag.clearForTesting();
    rag.addDocument("existing atomic load sentinel");
    const std::string corruptDbPath = "rag_corrupt_v2_test.bin";
    {
        std::ofstream corrupt(corruptDbPath, std::ios::binary);
        const char magic[8] = {'N','P','P','R','A','G','2','\0'};
        const uint32_t version = 2;
        const uint64_t count = 1;
        const uint64_t textLen = 64;
        corrupt.write(magic, sizeof(magic));
        corrupt.write(reinterpret_cast<const char*>(&version), sizeof(version));
        corrupt.write(reinterpret_cast<const char*>(&count), sizeof(count));
        corrupt.write(reinterpret_cast<const char*>(&textLen), sizeof(textLen));
        corrupt.write("truncated", 9);
    }
    rag.loadDatabase(corruptDbPath);
    const std::string afterCorruptLoad = rag.retrieveContext("atomic load sentinel", 1);
    ok &= check(afterCorruptLoad.find("existing atomic load sentinel") != std::string::npos,
                "corrupt database load leaves the existing index unchanged");
    std::remove(corruptDbPath.c_str());

    rag.clearForTesting();
    rag.addDocument("oversized load sentinel");
    const std::string oversizedDbPath = "rag_oversized_v2_test.bin";
    {
        std::ofstream oversized(oversizedDbPath, std::ios::binary);
        const char magic[8] = {'N','P','P','R','A','G','2','\0'};
        const uint32_t version = 2;
        const uint64_t count = 1;
        const uint64_t textLen = 64ULL * 1024ULL * 1024ULL;
        oversized.write(magic, sizeof(magic));
        oversized.write(reinterpret_cast<const char*>(&version), sizeof(version));
        oversized.write(reinterpret_cast<const char*>(&count), sizeof(count));
        oversized.write(reinterpret_cast<const char*>(&textLen), sizeof(textLen));
    }
    rag.loadDatabase(oversizedDbPath);
    const std::string afterOversizedLoad = rag.retrieveContext("oversized load sentinel", 1);
    ok &= check(afterOversizedLoad.find("oversized load sentinel") != std::string::npos,
                "oversized persisted fields are rejected without replacing the active index");
    std::remove(oversizedDbPath.c_str());

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
    const std::string overlapCore =
        "authentication middleware validates bearer token request response session";
    rag.addDocument(overlapCore + " primary chunk detail", "src/auth.cpp", "cpp");
    rag.addDocument(overlapCore + " overlapping chunk detail", "src/auth.cpp", "cpp");
    rag.addDocument("authentication audit logger records security event", "src/audit.cpp", "cpp");
    const std::string diversified = rag.retrieveContext("authentication bearer token", 2);
    const bool hasPrimary = diversified.find("primary chunk detail") != std::string::npos;
    const bool hasOverlap = diversified.find("overlapping chunk detail") != std::string::npos;
    ok &= check(hasPrimary != hasOverlap,
                "near-duplicate chunks from the same source consume only one result slot");
    ok &= check(diversified.find("authentication audit logger") != std::string::npos,
                "deduplicated retrieval fills topK with a distinct relevant result");

    rag.clearForTesting();
    const std::string utf8BoundaryDocument =
        std::string(1199, 'x') + "\xC5\xBC" +
        " utf8 boundary marker searchable token " + std::string(300, 'y');
    rag.addDocument(utf8BoundaryDocument);
    const std::string utf8Chunked = rag.retrieveContext("boundary marker searchable token", 1);
    ok &= check(utf8Chunked.find("\xC5\xBC") != std::string::npos,
                "UTF-8 multibyte characters remain intact at chunk boundaries");
    ok &= check(utf8Chunked.find("boundary marker searchable token") != std::string::npos,
                "UTF-8-safe chunking preserves retrieval content");

    rag.clearForTesting();
    if (!ok) return 1;
    std::cout << "RAG retrieval tests passed.\n";
    return 0;
}