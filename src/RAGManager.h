#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <map>

class RAGManager {
public:
    static RAGManager& getInstance() {
        static RAGManager instance;
        return instance;
    }

    // Dodaje nową wiedzę do bazy wektorowej (samoistne rekurencyjne uczenie się)
    void addDocument(const std::string& text);

    // Wyszukuje najbardziej podobne fragmenty z bazy wiedzy
    std::string retrieveContext(const std::string& query, int topK = 3);

    // Zapis/Odczyt na dysk (trwała pamięć)
    void saveDatabase(const std::string& dbPath);
    void loadDatabase(const std::string& dbPath);

private:
    RAGManager() = default;
    ~RAGManager() = default;
    RAGManager(const RAGManager&) = delete;
    RAGManager& operator=(const RAGManager&) = delete;

    struct Document {
        std::string text;
        std::vector<float> embedding;
    };

    std::vector<Document> knowledgeBase;
    std::mutex dbMutex;

    // Wbudowany lekki mechanizm generowania wektorów (Bag of Words / Hashing Trick)
    std::vector<float> computeEmbedding(const std::string& text);
    float cosineSimilarity(const std::vector<float>& vecA, const std::vector<float>& vecB);
};