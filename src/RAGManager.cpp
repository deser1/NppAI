#include "RAGManager.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

// Prosty hashowany wektor (Hashing Trick) dla lekkiego RAG-a bez bibliotek zewnętrznych
const int VECTOR_DIM = 256;

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
    if (text.empty()) return;
    
    Document doc;
    doc.text = text;
    doc.embedding = computeEmbedding(text);
    
    std::lock_guard<std::mutex> lock(dbMutex);
    // Unikajmy identycznych duplikatów
    for (const auto& existing : knowledgeBase) {
        if (existing.text == text) return;
    }
    knowledgeBase.push_back(doc);
}

std::string RAGManager::retrieveContext(const std::string& query, int topK) {
    if (query.empty()) return "";
    
    std::vector<float> queryVec = computeEmbedding(query);
    
    std::lock_guard<std::mutex> lock(dbMutex);
    if (knowledgeBase.empty()) return "";

    std::vector<std::pair<float, std::string>> scores;
    for (const auto& doc : knowledgeBase) {
        float sim = cosineSimilarity(queryVec, doc.embedding);
        if (sim > 0.1f) { // próg odcięcia
            scores.push_back({sim, doc.text});
        }
    }

    // Sortowanie malejąco
    std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
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