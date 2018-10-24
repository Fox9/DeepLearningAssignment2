#include <iostream>
#include <math.h>

#include "randlib.h" 
#include "mnist/mnist.h"

using namespace std;

#define numOfInputNodes 785
#define numOfHiddenNodes 10
#define numOfOutputNodes 785

void randomizeWeightMatrixForHidden(float weights[numOfHiddenNodes][numOfInputNodes]) {
    for(int i = 0; i < numOfHiddenNodes; i++) {
        for(int j = 0; j < numOfInputNodes; j++) {
            weights[i][j] = rand_weight();
        }
    }
}

void randomizeWeightMatrixForOutPut(float weights[numOfOutputNodes][numOfHiddenNodes]) {
    for(int i = 0; i < numOfOutputNodes; i++) {
        for(int j = 0; j < numOfHiddenNodes; j++) {
            weights[i][j] = rand_weight();
        }
    }
}

void get_output_hidden(float hiddenLyaer[], int input[], float weights[numOfHiddenNodes][numOfInputNodes]) {
    
    for(int i = 0; i < numOfHiddenNodes; i++) {
        float resultOfMultiplication = 0;
        for(int j = 0; j < numOfInputNodes; j++) {
            resultOfMultiplication += input[j] * weights[i][j];
        }
        hiddenLyaer[i] = resultOfMultiplication;
    }
}

void get_output(float output[], int input[], float weights[numOfOutputNodes][numOfHiddenNodes]) {
    
    for(int i = 0; i < numOfOutputNodes; i++) {
        float resultOfMultiplication = 0;
        for(int j = 0; j < numOfHiddenNodes; j++) {
            resultOfMultiplication += input[j] * weights[i][j];
        }
        output[i] = resultOfMultiplication;
    }
}

void squash_output(float output[]) {
    
    for(int i = 0; i < numOfOutputNodes; i++) {
        output[i] = 1.0 / (1.0 + pow(M_E, -1 * output[i]));
        // printf("squashed output[%d] = %f\n", i, output[i]);
    }
}

void get_error_for_output(float errors[], float target[], float output[]) {
    for(int i = 0; i < numOfOutputNodes; i++) {
        errors[i] = (target[i] - output[i]) * output[i] * (1 - output[i]);
    }
}

void get_error_for_hidden_layer(float errorsOutput[], float errorsHidden[], float hiddenOutput[], float weightsOutput[numOfOutputNodes][numOfHiddenNodes]) {
    float resultOfMultiplication = 0;
    for(int i = 0; i < numOfHiddenNodes; i++) {
        for(int j = 0; j < numOfOutputNodes; j++) {
            resultOfMultiplication += errorsOutput[j] * weightsOutput[i][j];
        }
    }
    
    for(int i = 0; i < numOfHiddenNodes; i++) {
        errorsOutput[i] = hiddenOutput[i] * (1 - hiddenOutput[i]) * resultOfMultiplication;
    }
    
}

void update_weights_output(float learningRate, float outputs[], float errors[], float weights[numOfOutputNodes][numOfHiddenNodes]) {
    float deltaWeights[numOfHiddenNodes][numOfOutputNodes];
    for(int i = 0; i < numOfHiddenNodes; i++) {
        for(int j = 0; j < numOfOutputNodes; j++) {
            deltaWeights[i][j] = learningRate  * outputs[i] * errors[j];
            weights[i][j] += deltaWeights[i][j];
        }
    }
}


void update_weights_hidden(float learningRate, float outputs[], float errors[], float weights[numOfHiddenNodes][numOfInputNodes]) {
    float deltaWeights[numOfInputNodes][numOfHiddenNodes];
    for(int i = 0; i < numOfInputNodes; i++) {
        for(int j = 0; j < numOfHiddenNodes; j++) {
            deltaWeights[i][j] = learningRate  * outputs[i] * errors[j];
            weights[i][j] += deltaWeights[i][j];
        }
    }
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
    
    float learningRate = 0.5;
    
    int inputNodes[numOfInputNodes];
    float hiddenNodes[numOfHiddenNodes];
    float outputNodes[numOfOutputNodes];
    
    float errorsHidden[numOfHiddenNodes];
    float errorsOutput[numOfOutputNodes];
    
    float weightsHidden[numOfHiddenNodes][numOfInputNodes];
    randomizeWeightMatrixForHidden(weightsHidden);
    
    float weightsOutput[numOfOutputNodes][numOfHiddenNodes];
    randomizeWeightMatrixForHidden(weightsOutput);
    
    for(int simulation = 0; simulation < 20; simulation++) {
        
        for(int picIndex = 0; picIndex < sizeData; picIndex++) {
            
            get_input(inputNodes, zData, picIndex, 0.3);
            
            get_output_hidden(hiddenNodes, inputNodes, weightsHidden);
            squash_output(hiddenNodes)
            
            get_output(outputNodes, hiddenNodes, weightsOutput);
            squash_output(outputNodes)
            
            get_error_for_output(errorsOutput, target, outputNodes);
            update_weights_output(learningRate, outputNodes, errorsOutput, weightsOutput);
            
            
            get_error_for_hidden_layer(errorsOutput, errorsHidden, hiddenNodes, weightsOutput);
            update_weights_hidden(learningRate, hiddenNodes, errorsOutput, weightsHidden);
            
        }
        
    }
    
    
    
    return 0;
}
