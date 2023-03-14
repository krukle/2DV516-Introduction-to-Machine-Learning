# Report
## Course: *2DV516*
## Student: *Christoffer Eid, ce223af*
## Assignment: *2*

### Excercise 2.2

Depends on what's meant by the best fit. We would need a test set to test the different polynomials MSE (Mean Square Error). Given the information we have now, seeking the lowest MSE will only lead to overfitting the function. I'd guess the best function in our case is one of the second to forth one, just because they look like they follow the data nicely, and dont seem to overfitted. Its not impossible though that the linear one is the best, though unlikely. 

### Excercise 3.6

The results are qualitively the same for the training accuracy (96-98%) while the test accuracy fluctuates more (94-99%). This because the beta is trained on the training set, making it more prone to fit that set, while the test set will leave more to chance. 

Changing the share between the two sets makes for interesting changes. Lowering the train set to 20% and increasing the test set to 80% makes the program prone to overfit the data, since the chance for diversity lessens with the loss of data. This results in sometimes perfect training accuracy while test accuracy lowers. I.e. a worse function. It also happens that the train set gets fluctuationg data, forcing some underfitting and resulting in low training accuracies.