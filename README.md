# Stone-Paper-Scissor!
Making use of electromyographic data obtained from human hands connected to an electromyography (EMG) sensor to train a neural network powered by Tensorflow Keras, to predict whether the inputted hand gesture is Stone/Paper/Scissor and then predict the opposing move to defeat the user

## Instructions
- Open the `.ipynb` file in Google Colab
- Create a [Kaggle](https://www.kaggle.com/) account and obtain a `kaggle.json` to use its API 
- Train the model by running all the blocks in the `.ipynb` file
- Use an instance of `stone_paper_scissor` class to play the game

## Using the `stone_paper_scissor` class

```python
class stone_paper_scissor:  
  def winningmove(self, inp):
    if(inp==0):
      return "I play paper"
    elif(inp==1):
      return "I play scissor"
    else:
      return "I play stone"

  """
  move = 0; if user inputted stone
  move = 1; if user inputted paper
  move = 2; if user inputted scissor
  """

  def play_stone(self):
    move = 0
    sample = test_stone.sample()
    # obtain an exemplar EMG data sample for the user specified move
    pred = np.argmax(model.predict(sample), axis=1)
    print(self.winningmove(pred))
    if(move==pred):
      print("I won yay!")
    else:
      print("Nice, you won!")

  def play_paper(self):
    move = 1
    sample = test_paper.sample()
    # obtain an exemplar EMG data sample for the user specified move
    pred = np.argmax(model.predict(sample), axis=1)
    print(self.winningmove(pred))
    if(move==pred):
      print("I won yay!")
    else:
      print("Nice, you won!")

  def play_scissor(self):
    move = 2
    sample = test_scissor.sample()
    # obtain an exemplar EMG data sample for the user specified move
    pred = np.argmax(model.predict(sample), axis=1)
    print(self.winningmove(pred))
    if(move==pred):
      print("I won yay!")
    else:
      print("Nice, you won!")

```
- Create an instance the in following way 
> `instance = stone_paper_scissor()`
- Make use of the `play_stone()`, `play_scissor()`, `play_paper()` methods to make a move
> `instance.play_stone()` to play the stone move
