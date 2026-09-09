#include "src/NppAIEngine.h"
#include <iostream>
#include <fstream>

int main() {
    NppAIEngine engine;
    if (engine.loadModel("models/NppAI-model-v1.nppai")) {
        std::cout << "Model zaladowany pomyslnie. Rozpoczynam generowanie...\n\n";
        std::string result = engine.generate("[USER]: Lista todo w Vue\n[AI]:\n", 4096, 
            [](char c, bool isThought) {
                // Pusty callback - NppAIEngine.cpp sam wypisuje do std::cout w celach testowych
            },
            [](int count) {
                // Symulacja usunięcia znaków z konsoli
                for(int i=0; i<count; i++) std::cout << "\b \b";
            }
        );
        std::ofstream out("output_test.txt");
        out << result;
        out.close();
        std::cout << "\n\nWygenerowano " << result.length() << " znakow.\n" << std::endl;
    } else {
        std::cerr << "Blad ladowania modelu!" << std::endl;
    }
    return 0;
}
