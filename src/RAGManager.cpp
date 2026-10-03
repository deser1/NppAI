#include "RAGManager.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <set>

// Prosty hashowany wektor (Hashing Trick) dla lekkiego RAG-a bez bibliotek zewnętrznych
const int VECTOR_DIM = 256;

namespace {
std::set<std::string> tokenizeUnique(const std::string& text) {
    std::set<std::string> tokens;
    std::string current;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else if (!current.empty()) {
            tokens.insert(current);
            current.clear();
        }
    }
    if (!current.empty())
        tokens.insert(current);
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
            if (existing.text == chunk) {
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
    if (query.empty()) return "";
    
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
    
    size_t size = knowledgeBase.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    
    for (const auto& doc : knowledgeBase) {
        size_t textLen = doc.text.size();
        outFile.write(reinterpret_cast<const char*>(&textLen), sizeof(textLen));
        outFile.write(doc.text.data(), textLen);
        
        outFile.write(reinterpret_cast<const char*>(doc.embedding.data()), VECTOR_DIM * sizeof(float));
    }
    outFile.close();
}

void RAGManager::loadDatabase(const std::string& dbPath) {
    std::lock_guard<std::mutex> lock(dbMutex);
    std::ifstream inFile(dbPath, std::ios::binary);
    if (!inFile.is_open()) return;
    
    size_t size = 0;
    if (!inFile.read(reinterpret_cast<char*>(&size), sizeof(size))) return;
    
    knowledgeBase.clear();
    for (size_t i = 0; i < size; ++i) {
        Document doc;
        size_t textLen = 0;
        inFile.read(reinterpret_cast<char*>(&textLen), sizeof(textLen));
        
        doc.text.resize(textLen);
        inFile.read(&doc.text[0], textLen);
        
        doc.embedding.resize(VECTOR_DIM);
        inFile.read(reinterpret_cast<char*>(doc.embedding.data()), VECTOR_DIM * sizeof(float));
        
        knowledgeBase.push_back(doc);
    }
    inFile.close();
}
#ifdef NPPAI_TESTING
void RAGManager::clearForTesting() {
    std::lock_guard<std::mutex> lock(dbMutex);
    knowledgeBase.clear();
}
#endif
