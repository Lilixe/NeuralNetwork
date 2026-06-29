#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <thread>

#include "tensor.h"

using namespace std;

static constexpr int INPUT_SIZE  = 784;
static constexpr int HIDDEN_SIZE = 64;
static constexpr int OUTPUT_SIZE = 10;

/**************************************************
 *                INITIALIZATION                  *
 **************************************************/

struct DigitImage
{
    int label;
    Tensor2D pixels;

    DigitImage(int l, int n, int m) : label(l), pixels(n, m) {}
};

vector<DigitImage> loadMNIST(const string &filename)
{
    ifstream file(filename);
    vector<DigitImage> dataset;

    if (!file.is_open())
    {
        cerr << "Failed to open file.\n";
        return dataset;
    }

    string line;
    getline(file, line); // skip header

    while (getline(file, line))
    {
        stringstream sstream(line);
        string value;

        if (!getline(sstream, value, ','))
            continue;
        int label;
        try { label = stoi(value); }
        catch (...) { continue; }

        DigitImage img(label, 1, INPUT_SIZE);
        int cpt = 0;
        while (getline(sstream, value, ',') && cpt < img.pixels.size())
        {
            try { img.pixels(0, cpt) = stof(value) / 255.0f; }
            catch (...) { img.pixels(0, cpt) = 0.0f; }
            ++cpt;
        }
        dataset.push_back(img);
    }

    file.close();
    return dataset;
}

void randomizeVect(vector<DigitImage> &dataset)
{
    random_device rd;
    mt19937 g(rd());
    shuffle(begin(dataset), end(dataset), g);
}

pair<vector<DigitImage>, vector<DigitImage>> splitData(vector<DigitImage> &dataset)
{
    int splitIndex = static_cast<int>(dataset.size() * 0.8);
    vector<DigitImage> train(dataset.begin(), dataset.begin() + splitIndex);
    vector<DigitImage> dev(dataset.begin() + splitIndex, dataset.end());
    return make_pair(train, dev);
}

tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> initParameters()
{
    Tensor2D b1(HIDDEN_SIZE, 1), b2(OUTPUT_SIZE, 1);
    Tensor2D w1(HIDDEN_SIZE, INPUT_SIZE), w2(OUTPUT_SIZE, HIDDEN_SIZE);

    random_device rd;
    mt19937 gen(rd());
    // He init: N(0, sqrt(2/fan_in)) — recommandé pour couches ReLU
    // Les biais restent à 0 (déjà zero-initialisés par Tensor2D)
    normal_distribution<float> dist_w1(0.0f, sqrtf(2.0f / INPUT_SIZE));
    normal_distribution<float> dist_w2(0.0f, sqrtf(2.0f / HIDDEN_SIZE));

    for (int i = 0; i < HIDDEN_SIZE; ++i)
        for (int j = 0; j < INPUT_SIZE; ++j)
            w1(i, j) = dist_w1(gen);
    for (int i = 0; i < OUTPUT_SIZE; ++i)
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            w2(i, j) = dist_w2(gen);

    return make_tuple(b1, b2, w1, w2);
}

pair<Tensor2D, Tensor2D> toMatrix(vector<DigitImage> &data)
{
    int cols = data.size();
    Tensor2D resultPxl(INPUT_SIZE, cols);
    Tensor2D resultLbl(1, cols);

    for (int i = 0; i < cols; ++i)
    {
        resultLbl(0, i) = data[i].label;
        for (int j = 0; j < INPUT_SIZE; ++j)
            resultPxl(j, i) = data[i].pixels(0, j);
    }
    return make_pair(resultLbl, resultPxl);
}

tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> initData(vector<DigitImage> trainData)
{
    randomizeVect(trainData);
    vector<DigitImage> devData;
    tie(trainData, devData) = splitData(trainData);

    Tensor2D trainLabels, trainPixels, devLabels, devPixels;
    tie(trainLabels, trainPixels) = toMatrix(trainData);
    tie(devLabels, devPixels)     = toMatrix(devData);

    return make_tuple(trainLabels, trainPixels, devLabels, devPixels);
}

/**************************************************
 *             FORWARD PROPAGATION                *
 **************************************************/

Tensor2D relu(const Tensor2D &t)
{
    Tensor2D result(t.rows(), t.cols());
    for (int i = 0; i < t.rows(); ++i)
        for (int j = 0; j < t.cols(); ++j)
            result(i, j) = (t(i, j) < 0.0f) ? 0.0f : t(i, j);
    return result;
}

Tensor2D softmax(const Tensor2D &t)
{
    Tensor2D result(t.rows(), t.cols());
    for (int j = 0; j < t.cols(); ++j)
    {
        float maxVal = t.maxCol(j);
        float sum = 0.0f;
        for (int i = 0; i < t.rows(); ++i)
        {
            result(i, j) = expf(t(i, j) - maxVal);
            sum += result(i, j);
        }
        for (int i = 0; i < result.rows(); ++i)
            result(i, j) /= sum;
    }
    return result;
}

Tensor2D forwardPropagationLayer(const Tensor2D &input, const Tensor2D &weights, const Tensor2D &biases)
{
    return weights.dot(input) + biases;
}

tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> ForwardPropagation(
    const Tensor2D &data, const Tensor2D &w1, const Tensor2D &b1,
    const Tensor2D &w2, const Tensor2D &b2)
{
    Tensor2D l1     = forwardPropagationLayer(data, w1, b1);
    Tensor2D l1relu = relu(l1);
    Tensor2D l2     = forwardPropagationLayer(l1relu, w2, b2);
    Tensor2D l2soft = softmax(l2);
    return make_tuple(l1, l1relu, l2, l2soft);
}

/**************************************************
 *             BACKWARD PROPAGATION               *
 **************************************************/

Tensor2D derivativeRelu(const Tensor2D &t, const Tensor2D &variator)
{
    Tensor2D result(t.rows(), t.cols());
    for (int i = 0; i < t.rows(); ++i)
        for (int j = 0; j < t.cols(); ++j)
            result(i, j) = (variator(i, j) > 0.0f) ? t(i, j) : 0.0f;
    return result;
}

Tensor2D oneHot(const Tensor2D &labels)
{
    Tensor2D result(labels.cols(), OUTPUT_SIZE);
    for (int i = 0; i < labels.cols(); ++i)
        result(i, static_cast<int>(labels(0, i))) = 1.0f;
    return result.transpose();
}

float crossEntropyLoss(const Tensor2D &predictions, const Tensor2D &labels)
{
    float loss = 0.0f;
    int m = predictions.cols();
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < predictions.rows(); ++j)
            loss += labels(j, i) * logf(predictions(j, i) + 1e-8f);
    return -loss / m;
}

// Returns (dw1, db1, dw2, db2, loss)
tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D, float> backwardPropagation(
    const Tensor2D &w2, const Tensor2D &l1, const Tensor2D &l1relu,
    const Tensor2D &pixels, const Tensor2D &softmaxOut, const Tensor2D &oneHotLabels)
{
    int m = softmaxOut.cols(); // number of samples

    float ce_loss = crossEntropyLoss(softmaxOut, oneHotLabels);

    Tensor2D dl2 = softmaxOut - oneHotLabels;
    Tensor2D dw2 = dl2.dot(l1relu.transpose()) * (1.0f / m);
    Tensor2D db2 = dl2.sumColumn() * (1.0f / m);

    Tensor2D dl1 = derivativeRelu(w2.transpose().dot(dl2), l1);
    Tensor2D dw1 = dl1.dot(pixels.transpose()) * (1.0f / m);
    Tensor2D db1 = dl1.sumColumn() * (1.0f / m);

    return make_tuple(dw1, db1, dw2, db2, ce_loss);
}

/**************************************************
 *         PARAMETER UPDATE & METRICS             *
 **************************************************/

float accuracy(const Tensor2D &predictions, const Tensor2D &labels)
{
    float correct = 0;
    for (int i = 0; i < predictions.cols(); ++i)
        if (predictions.maxColIdx(i) == static_cast<int>(labels(0, i)))
            ++correct;
    return (correct / predictions.cols()) * 100.0f;
}

// B: Permute les colonnes de pixels et labels avec le même ordre aléatoire
void shuffleData(Tensor2D &pixels, Tensor2D &labels, mt19937 &rng)
{
    int n = pixels.cols();
    vector<int> perm(n);
    iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end(), rng);

    Tensor2D newPix(pixels.rows(), n);
    Tensor2D newLbl(labels.rows(), n);
    for (int i = 0; i < n; ++i)
    {
        for (int r = 0; r < pixels.rows(); ++r)
            newPix(r, i) = pixels(r, perm[i]);
        for (int r = 0; r < labels.rows(); ++r)
            newLbl(r, i) = labels(r, perm[i]);
    }
    pixels = move(newPix);
    labels = move(newLbl);
}

// D: v = beta*v + grad  ;  w = w - alpha*v  (heavy-ball momentum)
void updateParameters(
    Tensor2D &w1, Tensor2D &b1, Tensor2D &w2, Tensor2D &b2,
    Tensor2D &vw1, Tensor2D &vb1, Tensor2D &vw2, Tensor2D &vb2,
    const Tensor2D &dw1, const Tensor2D &db1,
    const Tensor2D &dw2, const Tensor2D &db2,
    float alpha, float beta)
{
    vw1 = vw1 * beta + dw1;
    vb1 = vb1 * beta + db1;
    vw2 = vw2 * beta + dw2;
    vb2 = vb2 * beta + db2;

    w1 = w1 - vw1 * alpha;
    b1 = b1 - vb1 * alpha;
    w2 = w2 - vw2 * alpha;
    b2 = b2 - vb2 * alpha;
}

/**************************************************
 *  GRADIENT DESCENT — mini-batch, momentum, decay*
 **************************************************/

// Chaque thread calcule les gradients sur sa tranche du mini-batch.
// Les gradients sont moyennés avant un unique updateParameters par mini-batch.
tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> gradientDescent(
    Tensor2D trainLabels, Tensor2D trainPixels,
    const Tensor2D &devLabels, const Tensor2D &devPixels,
    int epochs, float alpha, float alpha_decay, float momentum_beta,
    int batch_size, int numThreads)
{
    Tensor2D b1, b2, w1, w2;
    tie(b1, b2, w1, w2) = initParameters();

    // D: Vecteurs vitesse initialisés à zéro
    Tensor2D vw1(HIDDEN_SIZE, INPUT_SIZE), vb1(HIDDEN_SIZE, 1);
    Tensor2D vw2(OUTPUT_SIZE, HIDDEN_SIZE), vb2(OUTPUT_SIZE, 1);

    int N = trainPixels.cols();
    vector<Tensor2D> dw1s(numThreads), db1s(numThreads);
    vector<Tensor2D> dw2s(numThreads), db2s(numThreads);

    mt19937 rng(random_device{}());

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // B: Mélange aléatoire des données en début d'epoch
        shuffleData(trainPixels, trainLabels, rng);

        // B: Boucle sur les mini-batches
        for (int bs = 0; bs < N; bs += batch_size)
        {
            int be          = min(bs + batch_size, N);
            int actual      = be - bs;
            int eff_threads = min(numThreads, actual);
            int sub         = actual / eff_threads;

            auto process = [&](int t, int abs_start, int abs_end)
            {
                Tensor2D bPix  = trainPixels.sliceCols(abs_start, abs_end);
                Tensor2D bLbl  = trainLabels.sliceCols(abs_start, abs_end);
                Tensor2D bOH   = oneHot(bLbl);
                Tensor2D l1, l1relu, l2, l2soft;
                tie(l1, l1relu, l2, l2soft) = ForwardPropagation(bPix, w1, b1, w2, b2);
                float loss;
                tie(dw1s[t], db1s[t], dw2s[t], db2s[t], loss) =
                    backwardPropagation(w2, l1, l1relu, bPix, l2soft, bOH);
            };

            vector<thread> threads;
            threads.reserve(eff_threads);
            for (int t = 0; t < eff_threads; ++t)
            {
                int ts = bs + t * sub;
                int te = (t == eff_threads - 1) ? be : bs + (t + 1) * sub;
                threads.emplace_back(process, t, ts, te);
            }
            for (auto &th : threads)
                th.join();

            // Moyenne des gradients entre threads
            Tensor2D dw1_avg = dw1s[0], db1_avg = db1s[0];
            Tensor2D dw2_avg = dw2s[0], db2_avg = db2s[0];
            for (int t = 1; t < eff_threads; ++t)
            {
                dw1_avg = dw1_avg + dw1s[t];
                db1_avg = db1_avg + db1s[t];
                dw2_avg = dw2_avg + dw2s[t];
                db2_avg = db2_avg + db2s[t];
            }
            float inv = 1.0f / eff_threads;

            // D: Mise à jour avec momentum
            updateParameters(w1, b1, w2, b2, vw1, vb1, vw2, vb2,
                             dw1_avg * inv, db1_avg * inv,
                             dw2_avg * inv, db2_avg * inv,
                             alpha, momentum_beta);
        }

        // F: Décroissance exponentielle du learning rate
        alpha *= alpha_decay;

        // C: Rapport toutes les 10 epochs (et à la dernière)
        if (epoch % 10 == 0 || epoch == epochs - 1)
        {
            Tensor2D l1d, l1rd, l2d, devPreds;
            tie(l1d, l1rd, l2d, devPreds) = ForwardPropagation(devPixels, w1, b1, w2, b2);
            float devLoss = crossEntropyLoss(devPreds, oneHot(devLabels));
            float devAcc  = accuracy(devPreds, devLabels);
            cout << "Epoch " << setw(3) << epoch
                 << "  |  loss: "    << fixed      << setprecision(4) << devLoss
                 << "  |  dev_acc: " << fixed      << setprecision(2) << devAcc  << "%"
                 << "  |  lr: "      << scientific << setprecision(3) << alpha   << "\n";
        }
    }

    return make_tuple(b1, b2, w1, w2);
}

/**************************************************
 *         SAVE / LOAD PARAMETERS                 *
 **************************************************/

void saveParameters(const Tensor2D &b1, const Tensor2D &b2,
                    const Tensor2D &w1, const Tensor2D &w2, const string &filename)
{
    ofstream output_file(filename, ios::binary);
    if (!output_file.is_open())
    {
        cerr << "Failed to open file for saving parameters.\n";
        return;
    }
    cout << "Saving parameters to " << filename << "...\n";

    auto saveTensor = [&](const Tensor2D &tensor)
    {
        for (int i = 0; i < tensor.rows(); ++i)
            for (int j = 0; j < tensor.cols(); ++j)
            {
                float value = tensor(i, j);
                output_file.write(reinterpret_cast<const char *>(&value), sizeof(float));
            }
    };

    saveTensor(b1);
    saveTensor(b2);
    saveTensor(w1);
    saveTensor(w2);
    cout << "Parameters saved successfully.\n";
}

tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> loadParameters(const string &filename)
{
    vector<Tensor2D> tensors = {
        Tensor2D(HIDDEN_SIZE, 1),           // b1
        Tensor2D(OUTPUT_SIZE, 1),           // b2
        Tensor2D(HIDDEN_SIZE, INPUT_SIZE),  // w1
        Tensor2D(OUTPUT_SIZE, HIDDEN_SIZE)  // w2
    };

    ifstream input_file(filename, ios::binary);
    if (!input_file.is_open())
    {
        cerr << "Failed to open file for loading parameters.\n";
        return make_tuple(tensors[0], tensors[1], tensors[2], tensors[3]);
    }

    cout << "Loading parameters from " << filename << "...\n";

    auto loadTensor = [&](Tensor2D &tensor)
    {
        for (int i = 0; i < tensor.rows(); ++i)
            for (int j = 0; j < tensor.cols(); ++j)
                if (!input_file.read(reinterpret_cast<char *>(&tensor(i, j)), sizeof(float)))
                    cerr << "Error while reading parameter file.\n";
    };

    for (auto &t : tensors)
        loadTensor(t);

    return make_tuple(tensors[0], tensors[1], tensors[2], tensors[3]);
}

/**************************************************
 *         TRAIN                                  *
 **************************************************/

tuple<Tensor2D, Tensor2D, Tensor2D, Tensor2D> trainNN(
    const string &trainFile, int epochs, float alpha, float alpha_decay,
    float momentum_beta, int batch_size, int numThreads)
{
    vector<DigitImage> trainData = loadMNIST(trainFile);

    Tensor2D trainLabels, trainPixels, devLabels, devPixels;
    tie(trainLabels, trainPixels, devLabels, devPixels) = initData(trainData);

    Tensor2D b1, b2, w1, w2;
    tie(b1, b2, w1, w2) = gradientDescent(
        trainLabels, trainPixels, devLabels, devPixels,
        epochs, alpha, alpha_decay, momentum_beta, batch_size, numThreads);

    cout << "\nExiting training.\n";
    saveParameters(b1, b2, w1, w2, "nn_parameters.txt");
    return make_tuple(b1, b2, w1, w2);
}

/**************************************************
 *         MAIN                                   *
 **************************************************/

int main()
{
    cout << "\n==============================\n";
    vector<DigitImage> testData = loadMNIST("../mnist_test.csv");
    Tensor2D testLabels, testPixels;
    tie(testLabels, testPixels) = toMatrix(testData);

    int   epochs        = 50;
    float alpha         = 0.05f;
    float alpha_decay   = 0.99f;  // F: lr *= decay chaque epoch
    float momentum_beta = 0.9f;   // D: coefficient de momentum
    int   batch_size    = 512;    // B: taille du mini-batch
    int   numThreads    = 4;

    Tensor2D b1, b2, w1, w2;
    tie(b1, b2, w1, w2) = trainNN("../mnist_train.csv", epochs, alpha,
                                   alpha_decay, momentum_beta, batch_size, numThreads);

    Tensor2D l1, l1relu, l2, predictions;
    tie(l1, l1relu, l2, predictions) = ForwardPropagation(testPixels, w1, b1, w2, b2);

    cout << "\n==============================\n";
    cout << "Final Accuracy on Test Set: " << accuracy(predictions, testLabels) << "%\n";
    cout << "==============================\n";

    return 0;
}
