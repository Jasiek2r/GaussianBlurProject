
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include "DataLoader.h"
#include "GaussianBlur.h"
#include "GaussianBlurSerial.h"

using namespace std;

const string filepath = "./Lena.png";
const int kernelSize = 150;
const int repetitions = 100;

void print_stats(vector<double> times) {
    // Obliczanie statystyk
    double sum = accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / repetitions;


    // Odchylenie standardowe z probki
    double squaredSum = 0.0;

    for (double t : times) {
        squaredSum += (t - mean) * (t - mean);
    }

    double stdDev = sqrt(squaredSum / (repetitions - 1));

    // Niepewnosc standardowa sredniej
    double uncertainty = stdDev / sqrt(repetitions);

    // Przyblizona niepewnosc rozszerzona 95%
    double expandedUncertainty = 1.984 * uncertainty;

    auto minMax = minmax_element(times.begin(), times.end());

    // Wyniki
    cout << fixed << setprecision(6);

    cout << "\n========== WYNIKI POMIAROW ==========" << endl;
    cout << "Liczba pomiarow: " << repetitions << endl;
    cout << "Sredni czas: " << mean << " ms" << endl;
    cout << "Odchylenie standardowe: " << stdDev << " ms" << endl;
    cout << "Niepewnosc standardowa sredniej: "
        << uncertainty << " ms" << endl;
    cout << "Niepewnosc rozszerzona (ok. 95%): "
        << expandedUncertainty << " ms" << endl;
    cout << "Minimalny czas: " << *minMax.first << " ms" << endl;
    cout << "Maksymalny czas: " << *minMax.second << " ms" << endl;

    cout << "====================================" << endl;
}

int main() {

    cout << "Maksymalna liczba watkow: "
        << omp_get_max_threads() << endl;

    DataLoader dataLoader;
    dataLoader.Load(filepath);

    if (!dataLoader.HasImage()) {
        cerr << "Nie udalo sie zaladowac obrazu!" << endl;
        return 1;
    }

    cv::Mat image = dataLoader.GetImage();
    cv::Mat result;

    cout << "Rozmiar obrazu: "
        << image.cols << "x" << image.rows << endl;

    // Rozgrzewka - pomiar nie jest uwzgledniany
    result = GaussianBlur::Process(image, kernelSize);

    vector<double> times;
    vector<double> times_serial;
    times.reserve(repetitions);
    times_serial.reserve(repetitions);

    // 100 pomiarow
    for (int i = 0; i < repetitions; i++) {
        double start = omp_get_wtime();

        result = GaussianBlur::Process(image, kernelSize);

        double end = omp_get_wtime();

        times.push_back((end - start) * 1000.0);
    }

    // 100 pomiarow
    for (int i = 0; i < repetitions; i++) {
        double start = omp_get_wtime();

        result = GaussianBlurSerial::Process(image, kernelSize);

        double end = omp_get_wtime();

        times_serial.push_back((end - start) * 1000.0);
    }

    print_stats(times);
    print_stats(times_serial);

    // Zapis ostatniego wyniku
    if (!cv::imwrite("Lena_rozmyta.png", result)) {
        cerr << "Blad zapisu obrazu!" << endl;
        return 1;
    }

    cout << "Wynik zapisany: Lena_rozmyta.png" << endl;

    return 0;
}
