#pragma once
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include <unordered_set>

class RAGManager {
public:
    static RAGManager& getInstance() {
        static RAGManager instance;
        return instance;
    }

    // Dodaje nową wiedzę do bazy wektorowej (samoistne rekurencyjne uczenie się)
    void addDocument(const std::string& text);
    void addDocument(const std::string& text, const std::string& source, const std::string& language);
    void updateSource(const std::string& text, const std::string& source, const std::string& language);
    size_t indexRepository(const std::string& rootPath);

    // Wyszukuje najbardziej podobne fragmenty z bazy wiedzy
    std::string retrieveContext(const std::string& query, int topK = 3);
    std::string retrieveContext(const std::string& query, int topK,
                                const std::string& sourceFilter,
                                const std::string& languageFilter);

    // Zapis/Odczyt na dysk (trwała pamięć)
    void saveDatabase(const std::string& dbPath);
    void loadDatabase(const std::string& dbPath);

#ifdef NPPAI_TESTING
    void clearForTesting();
#endif

private:
    RAGManager() = default;
    ~RAGManager() = default;
    RAGManager(const RAGManager&) = delete;
    RAGManager& operator=(const RAGManager&) = delete;

    struct Document {
        std::string text;
        std::vector<float> embedding;
        std::string source;
        std::string language;
    };

    std::vector<Document> knowledgeBase;
    std::unordered_set<std::string> documentKeys;
    std::mutex dbMutex;

    static std::string makeDocumentKey(const std::string& text,
                                       const std::string& source,
                                       const std::string& language);

    // Wbudowany lekki mechanizm generowania wektorów (Bag of Words / Hashing Trick)
    std::vector<float> computeEmbedding(const std::string& text);
    float cosineSimilarity(const std::vector<float>& vecA, const std::vector<float>& vecB);
};