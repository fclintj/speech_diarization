# Deep Learning Diarization 

## Project Description 
The speech diarization project obtains recorded speech with two speakers (each speaking into their own microphone) and returns an output file with the time intervals that they were speaking.

## Prerequisites
`matplotlib`  
`scipy`  
`sounddevice`  


## Getting Set-up
The training data should be placed into the media folder:  
`deep-diarization/media/TextGrids/`  
`deep-diarization/media/Sound Files/`  
etc. These files are listed in the git ignore and should never be committed into the repository because they are huge.

## Running
To run the training process, open the `main.py` file, and adjust the parameters as needed and run the script. Eventually, all parameters will be adjustable from the command line. 

To run the diarization method, set the `training_flag` to `False`. This will skip the training process and write the textgrid files to disk.

## Developing Diarization Methods
The folder `diarization_methods` should be the place to add new diarization methods. Each method should be a class that inherits from the `DiarizationBaseClass`. An example class, that does nothing but print out a statement in each method, is provided as a reference.

After a diarization method has been created, change the `diarization_method` parameter found in `main.py` to use your method.

There are also class and method comments inside of `DiarizationBaseClass` that will be worth reading, as it explains much more about how to use it. 

## Testing
Unit tests are a great way to help debug. Writing a unit test is easy with the nosetests framework. 

All you do is: 
1. Create a `TestSomething.py` (must start with 'test') in the tests folder. 
2. Add methods into it to test each function of your class or method. Test methods names should start with 'test'. 
3. Test methods should also make liberal use of `assert` statements. These are used to say 'If this statement is false, the test should fail'. eg `assert x == 4`
4. Then, in the terminal, `cd` to the tests folder and run `nosetests`. You might have to install it first, use pip or apt. nosetests with then go through all the tests within the tests folder and show you which ones pass and which ones fail.
5. Eat a taco and enjoy how easy that is.

## Authors
* Clint Ferrin
* Madi Mickelsen
* Daniel Mortensen
* Josh Vanfleet
