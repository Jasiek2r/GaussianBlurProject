#include <iostream>
#include <omp.h>
#include <string>
#include "DataLoader.h"
#include "GaussianBlur.h"

using namespace std;

int main() {
    // Sprawdzenie, czy OpenMP jest włączone
#ifndef _OPENMP
    std::cout << "OpenMP NIE jest wlaczone we wlasciwosciach projektu!" << std::endl;
    return 1;
#endif

    std::cout << "Maksymalna liczba watkow: " << omp_get_max_threads() << std::endl;

    const string filepath = "./Lena.png";
    const int kernelSize = 150;
    
    DataLoader dataLoader;
    dataLoader.Load(filepath);

    if (dataLoader.HasImage()) {
        cv::Mat image = dataLoader.GetImage();
        
        // Pomiar czasu rozpoczęcia
        double start = omp_get_wtime();

        // Wykonanie filtra Gaussa
        cv::Mat result = GaussianBlur::Process(image, kernelSize);

        // Pomiar czasu zakończenia
        double end = omp_get_wtime();

        // Obliczenie czasu wykonania
        double elapsed = end - start;

        std::cout << "Czas wykonania filtra Gaussa: "
            << elapsed << " sekund" << std::endl;
        
        cv::imwrite("Lena_rozmyta.png", result);
    }

    return 0;
}