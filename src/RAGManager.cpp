#include "RAGManager.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <set>
#include <cstdint>

// Prosty hashowany wektor (Hashing Trick) dla lekkiego RAG-a bez bibliotek zewnętrznych
const int VECTOR_DIM = 256;

namespace {
std::set<std::string> tokenizeUnique(const std::string& text) {
    std::set<std::string> tokens;
    std::string identifier;

    auto addIdentifier = [&]() {
        if (identifier.empty())
            return;

        std::string normalized;
        std::string part;
        for (size_t i = 0; i < identifier.size(); ++i) {
            const unsigned char uc = static_cast<unsigned char>(identifier[i]);
            const bool previousLowerOrDigit = i > 0 &&
                (std::islower(static_cast<unsigned char>(identifier[i - 1])) ||
                 std::isdigit(static_cast<unsigned char>(identifier[i - 1])));
            const bool acronymWordBoundary = i > 0 && i + 1 < identifier.size() &&
                std::isupper(static_cast<unsigned char>(identifier[i - 1])) &&
                std::isupper(uc) &&
                std::islower(static_cast<unsigned char>(identifier[i + 1]));
            const bool upperBoundary = std::isupper(uc) &&
                (previousLowerOrDigit || acronymWordBoundary);

            if (identifier[i] == '_' || upperBoundary) {
                if (!part.empty()) {
                    tokens.insert(part);
                    part.clear();
                }
                if (identifier[i] == '_') {
                    normalized += '_';
                    continue;
                }
            }

            const char lower = static_cast<char>(std::tolower(uc));
            normalized += lower;
            part += lower;
        }
        if (!part.empty())
            tokens.insert(part);
        if (!normalized.empty())
            tokens.insert(normalized);
        identifier.clear();
    };

    for (char c : text) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '_')
            identifier += c;
        else
            addIdentifier();
    }
    addIdentifier();
    return tokens;
}

float lexicalOverlap(const std::set<std::string>& queryTokens, const std::string& text) {
    if (queryTokens.empty())
        return 0.0f;
    const auto documentTokens = tokenizeUnique(text);
    size_t matches = 0;
    for (const auto& token : queryTokens) {
        if (documentTokens.count(token) != 0)
            ++matches;
    }
    return static_cast<float>(matches) / static_cast<float>(queryTokens.size());
}
}

std::vector<float> RAGManager::computeEmbedding(const std::string& text) {
    std::vector<float> vec(VECTOR_DIM, 0.0f);
    
    // Prosty tokenizator (dzielenie po spacjach i znakach interpunkcyjnych)
    std::string current_word = "";
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current_word += static_cast<char>(std::tolower(c));
        } else if (!current_word.empty()) {
            // Hash DJB2a
            unsigned long hash = 5381;
            for (char wc : current_word) {
                hash = ((hash << 5) + hash) + wc;
            }
            int index = hash % VECTOR_DIM;
            vec[index] += 1.0f;
            current_word = "";
        }
    }
    if (!current_word.empty()) {
        unsigned long hash = 5381;
        for (char wc : current_word) {
            hash = ((hash << 5) + hash) + wc;
        }
        int index = hash % VECTOR_DIM;
        vec[index] += 1.0f;
    }

    // Normalizacja L2
    float norm = 0.0f;
    for (float val : vec) {
        norm += val * val;
    }
    if (norm > 0.0f) {
        norm = std::sqrt(norm);
        for (float& val : vec) {
            val /= norm;
        }
    }
    return vec;
}

float RAGManager::cosineSimilarity(const std::vector<float>& vecA, const std::vector<float>& vecB) {
    float dotProduct = 0.0f;
    for (size_t i = 0; i < VECTOR_DIM; ++i) {
        dotProduct += vecA[i] * vecB[i];
    }
    return dotProduct;
}

void RAGManager::addDocument(const std::string& text) {
    addDocument(text, "", "");
}

void RAGManager::addDocument(const std::string& text, const std::string& source, const std::string& language) {
    if (text.empty()) return;

    constexpr size_t CHUNK_SIZE = 1200;
    constexpr size_t CHUNK_OVERLAP = 200;

    std::vector<std::string> chunks;
    if (text.size() <= CHUNK_SIZE) {
        chunks.push_back(text);
    } else {
        size_t start = 0;
        while (start < text.size()) {
            size_t end = std::min(start + CHUNK_SIZE, text.size());
            if (end < text.size()) {
                const size_t newline = text.rfind('\n', end);
                if (newline != std::string::npos && newline > start + CHUNK_SIZE / 2)
                    end = newline + 1;
            }
            chunks.push_back(text.substr(start, end - start));
            if (end == text.size())
                break;
            start = end > CHUNK_OVERLAP ? end - CHUNK_OVERLAP : end;
        }
    }

    std::lock_guard<std::mutex> lock(dbMutex);
    for (const auto& chunk : chunks) {
        bool duplicate = false;
        for (const auto& existing : knowledgeBase) {
            if (existing.text == chunk && existing.source == source &&
                existing.language == language) {
                duplicate = true;
                break;
            }
        }
        if (duplicate)
            continue;

        Document doc;
        doc.text = chunk;
        doc.embedding = computeEmbedding(chunk);
        doc.source = source;
        doc.language = language;
        knowledgeBase.push_back(std::move(doc));
    }
}

std::string RAGManager::retrieveContext(const std::string& query, int topK) {
    return retrieveContext(query, topK, "", "");
}

std::string RAGManager::retrieveContext(const std::string& query, int topK,
                                        const std::string& sourceFilter,
                                        const std::string& languageFilter) {
    if (query.empty() || topK <= 0) return "";
    
    std::vector<float> queryVec = computeEmbedding(query);
    const auto queryTokens = tokenizeUnique(query);
    
    std::lock_guard<std::mutex> lock(dbMutex);
    if (knowledgeBase.empty()) return "";

    std::vector<std::pair<float, std::string>> scores;
    for (const auto& doc : knowledgeBase) {
        if (!sourceFilter.empty() && doc.source != sourceFilter)
            continue;
        if (!languageFilter.empty() && doc.language != languageFilter)
            continue;
        const float cosine = cosineSimilarity(queryVec, doc.embedding);
        const float lexical = lexicalOverlap(queryTokens, doc.text);
        const float score = cosine * 0.75f + lexical * 0.25f;
        if (score > 0.1f) { // próg odcięcia
            scores.push_back({score, doc.text});
        }
    }

    // Sortowanie malejąco; tekst rozstrzyga remisy deterministycznie.
    std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
        if (std::fabs(a.first - b.first) > 1e-6f)
            return a.first > b.first;
        return a.second < b.second;
    });

    std::string resultContext = "";
    int added = 0;
    for (const auto& score : scores) {
        if (added >= topK) break;
        resultContext += "--- Zapisany Kontekst RAG (Podobieństwo: " + std::to_string(score.first) + ") ---\n";
        resultContext += score.second + "\n\n";
        added++;
    }
    
    return resultContext;
}

void RAGManager::saveDatabase(const std::string& dbPath) {
    std::lock_guard<std::mutex> lock(dbMutex);
    std::ofstream outFile(dbPath, std::ios::binary);
    if (!outFile.is_open()) return;

    const char magic[8] = {'N','P','P','R','A','G','2','\0'};
    const uint32_t version = 2;
    const uint64_t size = static_cast<uint64_t>(knowledgeBase.size());
    outFile.write(magic, sizeof(magic));
    outFile.write(reinterpret_cast<const char*>(&version), sizeof(version));
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));

    for (const auto& doc : knowledgeBase) {
        const uint64_t textLen = static_cast<uint64_t>(doc.text.size());
        const uint64_t sourceLen = static_cast<uint64_t>(doc.source.size());
        const uint64_t languageLen = static_cast<uint64_t>(doc.language.size());
        outFile.write(reinterpret_cast<const char*>(&textLen), sizeof(textLen));
        outFile.write(doc.text.data(), static_cast<std::streamsize>(textLen));
        outFile.write(reinterpret_cast<const char*>(&sourceLen), sizeof(sourceLen));
        outFile.write(doc.source.data(), static_cast<std::streamsize>(sourceLen));
        outFile.write(reinterpret_cast<const char*>(&languageLen), sizeof(languageLen));
        outFile.write(doc.language.data(), static_cast<std::streamsize>(languageLen));
        outFile.write(reinterpret_cast<const char*>(doc.embedding.data()), VECTOR_DIM * sizeof(float));
    }
}

void RAGManager::loadDatabase(const std::string& dbPath) {
    std::lock_guard<std::mutex> lock(dbMutex);
    std::ifstream inFile(dbPath, std::ios::binary);
    if (!inFile.is_open()) return;

    char magic[8] = {};
    inFile.read(magic, sizeof(magic));
    const bool isV2 = inFile && std::string(magic, 7) == "NPPRAG2";
    inFile.clear();
    inFile.seekg(0);

    std::vector<Document> loadedDocuments;
    if (isV2) {
        uint32_t version = 0;
        uint64_t size = 0;
        inFile.read(magic, sizeof(magic));
        if (!inFile.read(reinterpret_cast<char*>(&version), sizeof(version)) || version != 2 ||
            !inFile.read(reinterpret_cast<char*>(&size), sizeof(size)))
            return;

        for (uint64_t i = 0; i < size; ++i) {
            Document doc;
            uint64_t textLen = 0, sourceLen = 0, languageLen = 0;
            if (!inFile.read(reinterpret_cast<char*>(&textLen), sizeof(textLen))) return;
            doc.text.resize(static_cast<size_t>(textLen));
            if (textLen && !inFile.read(&doc.text[0], static_cast<std::streamsize>(textLen))) return;
            if (!inFile.read(reinterpret_cast<char*>(&sourceLen), sizeof(sourceLen))) return;
            doc.source.resize(static_cast<size_t>(sourceLen));
            if (sourceLen && !inFile.read(&doc.source[0], static_cast<std::streamsize>(sourceLen))) return;
            if (!inFile.read(reinterpret_cast<char*>(&languageLen), sizeof(languageLen))) return;
            doc.language.resize(static_cast<size_t>(languageLen));
            if (languageLen && !inFile.read(&doc.language[0], static_cast<std::streamsize>(languageLen))) return;
            doc.embedding.resize(VECTOR_DIM);
            if (!inFile.read(reinterpret_cast<char*>(doc.embedding.data()), VECTOR_DIM * sizeof(float))) return;
            loadedDocuments.push_back(std::move(doc));
        }
        knowledgeBase = std::move(loadedDocuments);
        return;
    }

    size_t size = 0;
    if (!inFile.read(reinterpret_cast<char*>(&size), sizeof(size))) return;
    for (size_t i = 0; i < size; ++i) {
        Document doc;
        size_t textLen = 0;
        if (!inFile.read(reinterpret_cast<char*>(&textLen), sizeof(textLen))) return;
        doc.text.resize(textLen);
        if (textLen && !inFile.read(&doc.text[0], static_cast<std::streamsize>(textLen))) return;
        doc.embedding.resize(VECTOR_DIM);
        if (!inFile.read(reinterpret_cast<char*>(doc.embedding.data()), VECTOR_DIM * sizeof(float))) return;
        loadedDocuments.push_back(std::move(doc));
    }
    knowledgeBase = std::move(loadedDocuments);
}
#ifdef NPPAI_TESTING
void RAGManager::clearForTesting() {
    std::lock_guard<std::mutex> lock(dbMutex);
    knowledgeBase.clear();
}
#endif
