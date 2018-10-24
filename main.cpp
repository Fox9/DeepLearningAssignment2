#include <iostream>

#include "randlib.h" 
#include "mnist/mnist.h"

using namespace std;

#define numOfInputNodes 785
#define numOfHiddenNodes 10
#define numOfOutputNodes 785

void randomizeWeightMatrixForHidden(float weights[numOfHiddenNodes][numOfInputNodes]) {
    for(int i = 0; i < numOfHiddenNodes; i++) {
        for(int j = 0; j < numOfInputNodes; j++) {
            matrix[i][j] = rand_weight();
        }
    }
}

void randomizeWeightMatrixForOutPut(float weights[numOfOutputNodes][numOfInputNodes]) {
    for(int i = 0; i < numOfOutputNodes; i++) {
        for(int j = 0; j < numOfInputNodes; j++) {
            matrix[i][j] = rand_weight();
        }
    }
}

void outputForHiddenLayer(float hiddenLyaer[], int input[], float weights[numOfHiddenNodes][numOfInputNodes]) {
    
    for(int i = 0; i < numOfHiddenNodes; i++) {
        float resultOfMultiplication = 0
        for(int j = 0; j < numOfInputNodes; j++) {
            resultOfMultiplication += input[j] * weights[i][j]
        }
        hiddenLyaer[i] = resultOfMultiplication;
    }
}

void outputForHiddenLayer(float output[], int input[], float weights[numOfOutputNodes][numOfInputNodes]) {
    
    for(int i = 0; i < numOfOutputNodes; i++) {
        float resultOfMultiplication = 0
        for(int j = 0; j < numOfInputNodes; j++) {
            resultOfMultiplication += input[j] * weights[i][j]
        }
        hiddenLyaer[i] = resultOfMultiplication;
    }
}

void squash_output(float output[]) {
    
    for(int i = 0; i < numOutputNodes; i++) {
        output[i] = 1.0 / (1.0 + pow(M_E, -1 * output[i]));
        // printf("squashed output[%d] = %f\n", i, output[i]);
    }
}

void get_error_hidden_layer(float target) {
    
}

int main() {
    // --- an example for working with random numbers
    seed_randoms();
    
    float sampNoise = 0;
    
    // --- a simple example of how to set params from the command line
    if(argc == 2){ // if an argument is provided, it is SampleNoise
        sampNoise = atof(argv[1]);
        if (sampNoise < 0 || sampNoise > .5){
            printf("Error: sample noise should be between 0.0 and 0.5\n");
            return 0;
        }
    }
    
    mnist_data *zData;      // each image is 28x28 pixels
    unsigned int sizeData;  // depends on loadType
    int loadType = 0; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData, &sizeData, loadType)){
        printf("something went wrong loading data set\n");
        return -1;
    }
    
    float learningRate = 0.5
    
    int inputNodes[numOfInputNodes];
    float hiddenNodes[numOfHiddenNodes];
    float outputNodes[numOfOutputNodes];
    float errors[numOfOutputNodes];
    
    float weightsHidden[numOfHiddenNodes][numOfInputNodes];
    randomizeWeightMatrixForHidden(weightsHidden);
    
    float weightsOutput[numOfOutputNodes][numOfInputNodes];
    randomizeWeightMatrixForHidden(weightsOutput);
    
    for(int simulation = 0; simulation < 20; simulation++) {
        
        for(int picIndex = 0; picIndex < sizeData; picIndex++) {
            
            get_input(inputNodes, zData, picIndex, 0.3);
            
            outputForHiddenLayer(hiddenNodes, inputNodes, weightsHidden);
            squash_output(hiddenNodes)
            
            get_output(outputNodes, hiddenNodes, weightsOutput);
            squash_output(weightsOutput)
            
            get_error()
            
        }
    }
    
    
    
    return 0;
}
