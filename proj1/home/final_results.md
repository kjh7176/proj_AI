# Final Results

#### Provide the results from running the check_images.py for all three CNN model architectures

| # Total Images | 40 |
| --- | --- |
| # Dog Images | 30 |
| # Not-a-Dog Images | 10 |

| CNN Model Architecture: | % Not-a-Dog Correct | % Dogs Correct | % Breeds Correct | % Match Labels | time |
| --- | --- | --- | --- | --- | --- |
| ResNet | 90% | **100%** | 90% | 82.5% | 0:0:5 |
| AlexNet | **100%** | **100%** | 80% | 75% | **0:0:3** |
| VGG | **100%** | **100%** | **93.33%** | **87.5%** | 0:0:33 |

#### Compare these results to the ones your program produced when you ran run_models_batch.sh (or run_models_batch_hints.sh) in the Printing Results section

The results printed out above is exactly same with the given results in the previous instruction.

#### Discuss how the check_images.py addressed the four primary objectives of this Project

1. **Correctly identify which pet images are of dogs (even if breed is misclassified) and which pet images aren't of dogs.**
In this program, we call the classifier function with arguments of file path and CNN model name. And then it returns the classified label.
It's correct if that label exists in the 'dognames.txt' which is a collection of breed names of dogs.

2. **Correctly classify the breed of dog, for the images that are of dogs.**
This program compares the title name of each image file and the classified label to check if the result of classification is correct.

3. **Determine which CNN model architecture (ResNet, AlexNet, or VGG), "best" achieve the objectives 1 and 2.**
For objective 1, Alexnet and VGG both performed perfectly at the rate of 100% accuracy in terms of identifying dogs or not.
For objective 2, VGG showed the best accuracy with 93.33% among three of them in breed classification.

4. **Consider the time resources required to best achieve objectives 1 and 2, and determine if an alternative solution would have given a "good enough" result, given the amount of time each of the algorithms take to run.**
AlexNet model perfectly identified dogs and required the shortest time.
ResNet model took the longest time to perform and it's less accurate to identify dogs than AlexNet model even though it can classify breed better.
Therefore, I think AlexNet model is good enough to be an alternative solution but ResNet model is not.
