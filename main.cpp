#include <iostream>
#include <math.h>

#include "randlib.h" 
#include "mnist/mnist.h"

using namespace std;

#define numOfInputNodes 785
#define numOfHiddenNodes 11
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

void initTarget(float target[], int input[], int numberOnPicture) {
    target[0] = ((numberOnPicture % 2) == 0) ? 0 : 1; // odd or even number
    for(int i = 0; i < numOfOutputNodes; i++) {
        target[i] = (float) input[i];
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
    hiddenLyaer[numOfHiddenNodes - 1] = 1; //bias for hidden nodes
}

void get_output(float output[], float input[], float weights[numOfOutputNodes][numOfHiddenNodes]) {
    
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
    for(int i = 0; i < numOfOutputNodes; i++) {
        for(int j = 0; j < numOfHiddenNodes; j++) {
            resultOfMultiplication += errorsOutput[j] * weightsOutput[i][j];
        }
    }
    
    for(int i = 0; i < numOfHiddenNodes; i++) {
        errorsOutput[i] = hiddenOutput[i] * (1 - hiddenOutput[i]) * resultOfMultiplication;
    }
    
}

float getAverageError(float error[]) {
    float errorsSum = 0;
    for (int i = 0; i < numOfOutputNodes; i++) {
        errorsSum += fabs(error[i]);
    }
    
    return (errorsSum / numOfOutputNodes);
}

void update_weights_output(float learningRate, float outputs[], float errors[], float weights[numOfOutputNodes][numOfHiddenNodes]) {
    float deltaWeights[numOfOutputNodes][numOfHiddenNodes];
    for(int i = 0; i < numOfOutputNodes; i++) {
        for(int j = 0; j < numOfHiddenNodes; j++) {
            deltaWeights[i][j] = learningRate  * outputs[i] * errors[j];
            weights[i][j] += deltaWeights[i][j];
        }
    }
}


void update_weights_hidden(float learningRate, float outputs[], float errors[], float weights[numOfHiddenNodes][numOfInputNodes]) {
    float deltaWeights[numOfHiddenNodes][numOfInputNodes];
    for(int i = 0; i < numOfHiddenNodes; i++) {
        for(int j = 0; j < numOfInputNodes; j++) {
            deltaWeights[i][j] = learningRate  * outputs[i] * errors[j];
            weights[i][j] += deltaWeights[i][j];
        }
    }
}

int main(int argc, char const *argv[]) {
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
    int loadType = 1; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData, &sizeData, loadType)){
        printf("something went wrong loading data set\n");
        return -1;
    }
    
    mnist_data *zTestingData;      // each image is 28x28 pixels
    unsigned int sizeTestingData;  // depends on loadType
    int loadTypeTesing = 2; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zTestingData, &sizeTestingData, loadTypeTesing)){
        printf("something went wrong loading data set\n");
        return -1;
    }
    
    float learningRate = 0.05;
    
    int inputNodes[numOfInputNodes];
    float hiddenNodes[numOfHiddenNodes];
    float outputNodes[numOfOutputNodes];
    
    float errorsHidden[numOfHiddenNodes];
    float errorsOutput[numOfOutputNodes];
    
    float target[numOfOutputNodes];
    
    float weightsHidden[numOfHiddenNodes][numOfInputNodes];
    randomizeWeightMatrixForHidden(weightsHidden);
    
    float weightsOutput[numOfOutputNodes][numOfHiddenNodes];
    randomizeWeightMatrixForOutPut(weightsOutput);
    
    for(int epoch = 0; epoch < 200; epoch++) {
        
        
        float resultError = 0;
        
        for(int picIndex = 0; picIndex < sizeTestingData; picIndex++) {
            get_input(inputNodes, zTestingData, picIndex, 0);
            
            initTarget(target, inputNodes, zTestingData[picIndex].label);
            
            get_output_hidden(hiddenNodes, inputNodes, weightsHidden);
            squash_output(hiddenNodes);
            
            get_output(outputNodes, hiddenNodes, weightsOutput);
            squash_output(outputNodes);
            
            get_error_for_output(errorsOutput, target, outputNodes);
            resultError += getAverageError(errorsOutput);
            
        }
        
        cout << (resultError / sizeTestingData) << ", ";
        
        for(int picIndex = 0; picIndex < sizeData; picIndex++) {
            
            get_input(inputNodes, zData, picIndex, 0);
            
            initTarget(target, inputNodes, zData[picIndex].label);
            
            get_output_hidden(hiddenNodes, inputNodes, weightsHidden);
            squash_output(hiddenNodes);
            
            get_output(outputNodes, hiddenNodes, weightsOutput);
            squash_output(outputNodes);
            
            get_error_for_output(errorsOutput, target, outputNodes);
            update_weights_output(learningRate, outputNodes, errorsOutput, weightsOutput);
            
            get_error_for_hidden_layer(errorsOutput, errorsHidden, hiddenNodes, weightsOutput);
            update_weights_hidden(learningRate, hiddenNodes, errorsHidden, weightsHidden);
        }
        
    }
    
    cout << endl;
    
    
    
    return 0;
}
