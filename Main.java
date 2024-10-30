/*
Assignment 2-2 for CSC 475 - handwritten digit recognition neural network implementation
Zoe Baker
10/21/24
*/
import java.util.*;
import java.io.*;

//Matrix class that handles both 1D and 2D Arrays with arbitrary parameters
class Matrix{
	public static int learningRate = 3;
	public static int epochSize = 30;
	public static int minibatchSize = 10;
	public double[][] matrix;
	public Matrix(double[]... rows){
		this.matrix = rows;
	}
	
	public double[][] getMatrix(){
		return this.matrix;
	}
	
	public double[] getArray(int i){
		return this.matrix[i];
	}
	
	public double getValue(int i, int j){
		return this.matrix[i][j];
	}
	
	public void setMatrix(double[][] rows){
		this.matrix = rows;
	}
	
	public void setArray(int i, double[] row){
		this.matrix[i] = row;
	}
	
	public void setValue(int i, int j, double value){
		this.matrix[i][j] = value;
	}
	
	// //A^L = sigmoid((weight vector ^ Layer) * (input vector ^ Layer-1) + (bias ^ Layer))
	// //this.matrix is previous values
	public Matrix forwardPropagate(Matrix weights, Matrix biases){	//layer indicates which layer we are on
		Matrix e1 = weights.multiplyByMatrix(this);
		Matrix e2 = e1.addMatrices(biases);
		Matrix nextValues = e2.performSigmoid();
		return nextValues;
	}
	
	//biasgradient^L = (W^L-1 * biasgradient^L-1)
	//schur A^L component multiplied over (1 - A^L)
	//this.matrix is actual values
	public Matrix getOutputBiasGrad(Matrix expectedValues){
		Matrix biasGradient = this.calculateError(expectedValues);
		return biasGradient;
	}
	
	//gradB^1 = (W^2^transpose * gradB^2) schur A^1 schur (1 - A^1)
	//this.matrix is layer values
	public Matrix getHiddenBiasGrad(Matrix finalWeights, Matrix prevBiasGrad){
		double[] ones = new double[this.matrix.length];
		Arrays.fill(ones, 1.0);
		Matrix matrixOfOnes = new Matrix(ones);
		matrixOfOnes = matrixOfOnes.transposeMatrix();
		Matrix e1 = finalWeights.transposeMatrix().multiplyByMatrix(prevBiasGrad);
		Matrix e2 = e1.performSchur(this);
		Matrix biasGradient = e2.performSchur(matrixOfOnes.subMatrices(this));
		return biasGradient;
	}
	
	//weightgradient^L = biasgradient^L * (A^L-1)^transposed
	//this.matrix is previous values
	public Matrix getWeightGrad(Matrix biasGradient){
		Matrix weightGradient = biasGradient.multiplyByMatrix(this.transposeMatrix());
		return weightGradient;
	}
	
	//bias = b-old - l/m(sum biasgradients)
	//weight = w-old - l/m(sum weightgradients)
	//this.matrix is old bias/weight
	public Matrix adjustWeightBias(Matrix ... gradients){
		Matrix sum = gradients[0].addMatrices(gradients[1]);
		if(gradients.length > 1){
			for(int i = 2; i < gradients.length; i++){
				sum = sum.addMatrices(gradients[i]);
			}
		}
		sum = sum.multiplyByConstant(learningRate/minibatchSize);
		Matrix adjustedVals = this.subMatrices(sum);
		return adjustedVals;
	}
	
	//Helper/math functions
	
	//transposes 2D matrix
	public Matrix transposeMatrix(){
		//switch length rows and cols from this.matrix
		//iterate this.matrix cols/transposed rows
		Matrix transposedMatrix = new Matrix(new double[this.matrix[0].length][this.matrix.length]);
		for(int i = 0; i < this.matrix.length; i++){
			for(int j = 0; j < this.matrix[0].length; j++){
				transposedMatrix.setValue(j,i,this.matrix[i][j]);
			}
		}
		return transposedMatrix;
	}
	
	//adds two matrices
	public Matrix addMatrices(Matrix matrixB){
		Matrix sumMatrix = new Matrix(new double[this.matrix.length][this.matrix[0].length]);
		for(int i = 0; i < this.matrix.length; i++){
			for(int j = 0; j < this.matrix[0].length; j++){
				sumMatrix.setValue(i,j,this.matrix[i][j] + matrixB.getValue(i,j));
			}
		}
		return sumMatrix;
	}
	
	//subtracts two matrices
	public Matrix subMatrices(Matrix matrixB){
		Matrix differenceMatrix = new Matrix(new double[this.matrix.length][this.matrix[0].length]);
		for(int i = 0; i < this.matrix.length; i++){
			for(int j = 0; j < this.matrix[0].length; j++){
				differenceMatrix.setValue(i,j,this.matrix[i][j] - matrixB.getValue(i,j));
			}
		}
		return differenceMatrix;
	}
	
	//finds schur factor for two matrices
	public Matrix performSchur(Matrix matrixB){
		Matrix productMatrix = new Matrix(new double[this.matrix.length][this.matrix[0].length]);
		for(int i = 0; i < this.matrix.length; i++){
			for(int j = 0; j < this.matrix[0].length; j++){
			productMatrix.setValue(i,j,this.matrix[i][j] * matrixB.getValue(i,j));
			}
		}
		return productMatrix;
	}
	
	//multiplies two matrices
	public Matrix multiplyByMatrix(Matrix matrixB){
		//matrixA = n x m | matrixB = m x p ~ prodMatrix = n x p
		//for each row in A, multiply by the column in B. This becomes[r][c] of prodMatrix
		Matrix prodMatrix = new Matrix(new double[this.matrix.length][matrixB.getArray(0).length]);
		for(int rowA = 0; rowA < this.matrix.length; rowA++){
			for(int colB = 0; colB < matrixB.getArray(0).length; colB++){
				double dotProduct = 0;
				for(int i = 0; i < this.matrix[0].length; i++){
					dotProduct += this.matrix[rowA][i] * matrixB.getValue(i, colB);
				}
				prodMatrix.setValue(rowA, colB, dotProduct);
			}
		}
		return prodMatrix;
	}
	
	//multiplies a matrix by an integer
	public Matrix multiplyByConstant(int num){
		Matrix product = new Matrix(this.matrix);
		for(int i = 0; i < this.matrix.length; i++){
			for(int j = 0; j < this.matrix[0].length; j++){
				product.setValue(i,j,this.matrix[i][j] * num);
			}
		}
		return product;
	}
	
	//performs sigmoid function on an array
	public Matrix performSigmoid(){
		Matrix sigmoidResult = new Matrix(this.matrix);
		for(int i = 0; i < this.matrix.length; i++){
			for(int j = 0; j < this.matrix[0].length; j++){
			sigmoidResult.setValue(i,j,1/(1 + Math.exp(-this.matrix[i][j])));
			}
		}
		return sigmoidResult;
	}
	
	//error = (A - expectedvalues) schur A schur (1 - A)
	public Matrix calculateError(Matrix expectedValues){
		Matrix error = new Matrix(new double[this.matrix.length]);
		double[] ones = new double[error.getArray(0).length];
		Arrays.fill(ones, 1.0);
		Matrix matrixOfOnes = new Matrix(ones);
		matrixOfOnes = matrixOfOnes.transposeMatrix();
		error = this.subMatrices(expectedValues);
		error = error.performSchur(this);
		error = error.performSchur(matrixOfOnes.subMatrices(this));
		return error;
	}

	//prints the given matrix in a human readable format
	public void printMatrix(){
		for(int i = 0; i < this.matrix.length; i++){	
			System.out.print("[");
			for(int j = 0; j < this.matrix[0].length; j++){
				double val = (double)Math.round(this.matrix[i][j] * 1000d) / 1000d;
				System.out.print(val + "\t");
			}
			System.out.println("]");
		}
		System.out.println();
	}
}

//Dataset class w/ objs representing X & Y
class Dataset{
	public Matrix X;
	public Matrix Y;
	public Dataset(Matrix x, Matrix y){
		this.X = x;
		this.Y = y;
	}
	
	//compares Y (m1) and finalValues (m2) and returns whether they are equivalent
	public static boolean compare(Matrix m1, Matrix m2){
		boolean equal = true;
		int target = Dataset.getIntY(m1);
		if(m2.getValue(target,0) != target){equal = false;}
		
		return equal;
	}
	
	//returns Y as an int intstead of a Matrix
	public static int getIntY(Matrix yMat){
		int largestIndex = 0;
		for(int i = 0; i < yMat.getMatrix().length; i++){
			if(yMat.getValue(i,0) > yMat.getValue(largestIndex,0)){largestIndex = i;}
		}
		return largestIndex;
	}	
	
}

//Options for Command Line Interface
class CLI{
	//creating file readers and writers
	
	
	// public File testFile = new File("mnist_test.csv");
	// public FileReader frtest = new FileReader(testFile);
	// public BufferedReader brtest = new BufferedReader(frtest);
	
	public static Matrix weights1 = new Matrix(
		new double[784], 
		new double[784], 
		new double[784], 
		new double[784], 
		new double[784], 
		new double[784], 
		new double[784], 
		new double[784], 
		new double[784], 
		new double[784], 
		new double[784], 
		new double[784], 
		new double[784], 
		new double[784], 
		new double[784]
	);
	public static Matrix weights2 = new Matrix(
		new double[15],
		new double[15],
		new double[15],
		new double[15],
		new double[15],
		new double[15],
		new double[15],
		new double[15],
		new double[15],
		new double[15]
	);
	public static Matrix biases1 = new Matrix(
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1]
	);
	public static Matrix biases2 = new Matrix(
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1],
		new double[1]
	);
	public static boolean useRandom = true;
	
	public CLI(){
		return;
	}
	
	//Option 1: trains the network on given data using randomly generated W/B from -1 to 1
	public static void trainNetwork() throws IOException{	//change to return a matrix of how many of each value were correct
		System.out.println("Training Neural Network . . .");
		File trainFile = new File("mnist_train.csv");
		FileReader fr = new FileReader(trainFile);
		BufferedReader br = new BufferedReader(fr);
	
		//Create dataset from training file
		String valuesString;
		int lineTracker = 0;
		Dataset[] data = new Dataset[60000];
		while((valuesString = br.readLine()) != null){
			//creating input (X)
			String xString = valuesString.substring(2);
			//creating list of doubles
			double[] xValues = new double[784];
			//converting string of values to array of values
			int strIterate = 0;
			int arrIterate = 0;
			while(strIterate < xString.length()){
				String currentNum = "";
				//create a string of the next number based on where the next "," is
				while(Character.isDigit(xString.charAt(strIterate))){
					currentNum += xString.charAt(strIterate);
					strIterate++;
					//check if at end of line
					if(strIterate >= xString.length()){break;}
				}
				double currentValue = Double.parseDouble(currentNum);
				xValues[arrIterate] = currentValue;
				//skip comma
				strIterate++;
				//move to next array slot
				arrIterate++;
			}
			//converting arry of values to Matrix of values
			Matrix X = new Matrix(xValues);
			X = X.transposeMatrix();
			
			//creating expected output (Y)
			//grabbing the value as a char
			int outputVal = Integer.valueOf(valuesString.substring(0,1));
			double[] yVals = new double[10];
			Arrays.fill(yVals, 0.0);
			yVals[outputVal] = 1.0;
			Matrix Y = new Matrix(yVals);
			Y = Y.transposeMatrix();
			
			data[lineTracker] = new Dataset(X, Y);
			lineTracker++;
		}
			
		//repurposing lineTracker variable for training
		lineTracker = 0;
		for(int epoch = 0; epoch < Matrix.epochSize; epoch++){
			//randomize training set if necessary
			if(useRandom){
				for(int i = 0; i < weights1.getMatrix().length; i++){
					for(int j = 0; j < weights1.getArray(0).length; j++){
						weights1.setValue(i,j,Math.random() * 2 - 1);
					}
				}
				for(int i = 0; i < biases1.getMatrix().length; i++){
					for(int j = 0; j < biases1.getArray(0).length; j++){
						biases1.setValue(i,j,Math.random() * 2 - 1);
					}
				}
				for(int i = 0; i < weights2.getMatrix().length; i++){
					for(int j = 0; j < weights2.getArray(0).length; j++){
						weights2.setValue(i,j,Math.random() * 2 - 1);
					}
				}
				for(int i = 0; i < biases2.getMatrix().length; i++){
					for(int j = 0; j < biases2.getArray(0).length; j++){
						biases2.setValue(i,j,Math.random() * 2 - 1);
					}
				}
			}
			//initializing gradient arrays for w/b adjustments
			Matrix[] outputBiases = new Matrix[Matrix.minibatchSize];
			Matrix[] hiddenBiases = new Matrix[Matrix.minibatchSize];
			Matrix[] outputWeights = new Matrix[Matrix.minibatchSize];
			Matrix[] hiddenWeights = new Matrix[Matrix.minibatchSize];
			
			
			//minibatch execution
			for(int minibatch = 0; minibatch < Matrix.minibatchSize; minibatch++){
				//forward
				Matrix hiddenValues = data[lineTracker].X.forwardPropagate(weights1, biases1);
				Matrix finalValues = hiddenValues.forwardPropagate(weights2, biases2);
				
				//back (adding to gradient arrays for w/b adjustments)
				outputBiases[minibatch] = finalValues.getOutputBiasGrad(data[lineTracker].Y);
				outputWeights[minibatch] = hiddenValues.getWeightGrad(outputBiases[minibatch]);
				hiddenBiases[minibatch] = hiddenValues.getHiddenBiasGrad(weights2, outputBiases[minibatch]);
				hiddenWeights[minibatch] = data[lineTracker].X.getWeightGrad(hiddenBiases[minibatch]);
				
				//point to next line
				lineTracker++;
				
			}
			//adjust weights and biases from minibatch
			biases2 = biases2.adjustWeightBias(outputBiases);
			weights2 = weights2.adjustWeightBias(outputWeights);
			biases1 = biases1.adjustWeightBias(hiddenBiases);
			weights1 = weights1.adjustWeightBias(hiddenWeights);
		}
		
		//Close resources
		br.close();
		fr.close();
		
		System.out.println("Neural Network successfully trained.");
	}
	
	//Option 2: loads a network with pre-trained W/B
	public static void loadNetwork() throws IOException{
		System.out.println("Loading saved network . . .");
		//set using random w/b to false since we are loading pre-saved values
		useRandom = false;
		File wFile1 = new File("mnist_weights1.csv");
		FileReader frw1 = new FileReader(wFile1);
		BufferedReader brw1 = new BufferedReader(frw1);
		
		File bFile1 = new File("mnist_biases1.csv");
		FileReader frb1 = new FileReader(bFile1);
		BufferedReader brb1 = new BufferedReader(frb1);
		
		File wFile2 = new File("mnist_weights2.csv");
		FileReader frw2 = new FileReader(wFile2);
		BufferedReader brw2 = new BufferedReader(frw2);
		
		File bFile2 = new File("mnist_biases2.csv");
		FileReader frb2 = new FileReader(bFile2);
		BufferedReader brb2 = new BufferedReader(frb2);
			
		//Create dataset from training file
		String xString;
		int lineTracker = 0;
		
		//bias 2
		while((xString = brb2.readLine()) != null){
			//creating list of doubles
			double[] xValues = new double[1];
			//converting string of values to array of values
			int strIterate = 0;
			int arrIterate = 0;
			while(strIterate < xString.length() - 1){
				String currentNum = "";
				//create a string of the next number based on where the next "," is
				while(!(xString.charAt(strIterate) == ',')){
					currentNum += xString.charAt(strIterate);
					strIterate++;
					if(strIterate >= xString.length() - 1){break;}
				}
				double currentValue = Double.parseDouble(currentNum);
				xValues[arrIterate] = currentValue;
				//skip comma
				strIterate++;
				//move to next array slot
				arrIterate++;
			}
			//converting arry of values to Matrix of values
			biases2.setArray(lineTracker, xValues);
			lineTracker++;
		}
		
		//bias 1
		xString = "";
		lineTracker = 0;
		while((xString = brb1.readLine()) != null){
			//creating list of doubles
			double[] xValues = new double[1];
			//converting string of values to array of values
			int strIterate = 0;
			int arrIterate = 0;
			while(strIterate < xString.length() - 1){
				String currentNum = "";
				//create a string of the next number based on where the next "," is
				while(!(xString.charAt(strIterate) == ',')){
					currentNum += xString.charAt(strIterate);
					strIterate++;
					if(strIterate >= xString.length() - 1){break;}
				}
				double currentValue = Double.parseDouble(currentNum);
				xValues[arrIterate] = currentValue;
				//skip comma
				strIterate++;
				//move to next array slot
				arrIterate++;
			}
			//converting arry of values to Matrix of values
			biases1.setArray(lineTracker, xValues);
			lineTracker++;
		}
		
		//weight 2
		xString = "";
		lineTracker = 0;
		while((xString = brw2.readLine()) != null){
			//creating list of doubles
			double[] xValues = new double[15];
			//converting string of values to array of values
			int strIterate = 0;
			int arrIterate = 0;
			while(strIterate < xString.length() - 1){
				String currentNum = "";
				//create a string of the next number based on where the next "," is
				while(!(xString.charAt(strIterate) == ',')){
					currentNum += xString.charAt(strIterate);
					strIterate++;
					if(strIterate >= xString.length() - 1){break;}
				}
				double currentValue = Double.parseDouble(currentNum);
				xValues[arrIterate] = currentValue;
				//skip comma
				strIterate++;
				//move to next array slot
				arrIterate++;
			}
			//converting arry of values to Matrix of values
			weights2.setArray(lineTracker, xValues);
			lineTracker++;
		}
		
		//weights 1
		xString = "";
		lineTracker = 0;
		while((xString = brw1.readLine()) != null){
			//creating list of doubles
			double[] xValues = new double[784];
			//converting string of values to array of values
			int strIterate = 0;
			int arrIterate = 0;
			while(strIterate < xString.length() - 1){
				String currentNum = "";
				//create a string of the next number based on where the next "," is
				while(!(xString.charAt(strIterate) == ',')){
					currentNum += xString.charAt(strIterate);
					strIterate++;
					if(strIterate >= xString.length() - 1){break;}
				}
				double currentValue = Double.parseDouble(currentNum);
				xValues[arrIterate] = currentValue;
				//skip comma
				strIterate++;
				//move to next array slot
				arrIterate++;
			}
			//converting arry of values to Matrix of values
			weights1.setArray(lineTracker, xValues);
			lineTracker++;
		}
		
		//Close resources
		brw1.close();
		frw1.close();
		brb1.close();
		frb1.close();
		brw2.close();
		frw2.close();
		brb2.close();
		frb2.close();
		
		System.out.println("Saved network successfully loaded.");
	}
	
	//Option 3: displays accuracy on training data after selecting either 1 or 2
	//Option 4: displays accuracy on testing data after selecting either 1 or 2
	public static void displayAccuracy(String decision) throws IOException{
		System.out.println("Fetching accuracy data . . .\n\n");
		//use decision to determin which dataset to use
		File file;
		int datasetSize;
		if(decision.equals("test")){
			file = new File("mnist_test.csv");
			datasetSize = 10000;
		}
		else{
			file = new File("mnist_train.csv");
			datasetSize = 60000;
		}
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		
		//Create dataset from file
		Dataset[] data = new Dataset[datasetSize];
		String valuesString;
		int lineTracker = 0;
		while((valuesString = br.readLine()) != null){
			//creating input (X)
			String xString = valuesString.substring(2);
			//creating list of doubles
			double[] xValues = new double[784];
			//converting string of values to array of values
			int strIterate = 0;
			int arrIterate = 0;
			while(strIterate < xString.length()){
				String currentNum = "";
				//create a string of the next number based on where the next "," is
				while(Character.isDigit(xString.charAt(strIterate))){
					currentNum += xString.charAt(strIterate);
					strIterate++;
					//check if at end of line
					if(strIterate >= xString.length()){break;}
				}
				double currentValue = Double.parseDouble(currentNum);
				xValues[arrIterate] = currentValue;
				//skip comma
				strIterate++;
				//move to next array slot
				arrIterate++;
			}
			//converting arry of values to Matrix of values
			Matrix X = new Matrix(xValues);
			X = X.transposeMatrix();
			
			//creating expected output (Y)
			//grabbing the value as a char
			int outputVal = Integer.valueOf(valuesString.substring(0,1));
			double[] yVals = new double[10];
			Arrays.fill(yVals, 0.0);
			yVals[outputVal] = 1.0;
			Matrix Y = new Matrix(yVals);
			Y = Y.transposeMatrix();
			
			data[lineTracker] = new Dataset(X, Y);
			lineTracker++;
		}

		//determining correct and classified array variables
		int[] correctList = trackAccuracy(data);
		int[] correct = new int[10];
		int[] classified = new int[10];
		Arrays.fill(classified, 0);
		for(int i = 0; i < data.length; i++){
			int yVal = Dataset.getIntY(data[i].Y);
			classified[yVal]++;
			if(correctList[i] == 1){correct[yVal]++;}
		}
		
		//display results
		int correctTotal = 0, classifiedTotal = 0;
		for(int i = 0; i < 5; i++){
			System.out.println("Digit " + i + ": " + correct[i] + "/" + classified[i]
				+ "\t\t" + "Digit " + ((int)i+5) + ": " + correct[i+5] + "/" + classified[i+5]
			);
			correctTotal += correct[i] + correct[i+5];
			classifiedTotal += classified[i] + classified[i+5];
		}
		double accuracy = (double)correctTotal / (double)classifiedTotal;
		//rounding
		accuracy = Math.round(accuracy * 100000.0) / 1000.0;
		System.out.println("\nAccuracy: " + correctTotal + "/" + classifiedTotal + " = " + accuracy + "%");
		
		//close resources
		fr.close();
		br.close();
	}
	
	//helper function - only tests the network with the directed matrix and no back propagation
	public static int[] trackAccuracy(Dataset[] data) throws IOException{
		//use decision to determin which dataset to use
		File file;
		if(data.length == 10000){
			file = new File("mnist_test.csv");
		}
		else{
			file = new File("mnist_train.csv");
		}
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		Scanner scanner = new Scanner(System.in);
		
		int[] correct = new int[data.length];
		int lineTracker = 0;
		for(int epoch = 0; epoch < Matrix.epochSize; epoch++){
			//minibatch execution
			for(int minibatch = 0; minibatch < Matrix.minibatchSize; minibatch++){
				//forward
				Matrix hiddenValues = data[lineTracker].X.forwardPropagate(weights1, biases1);
				Matrix finalValues = hiddenValues.forwardPropagate(weights2, biases2);
				//checks whether the value was correct. 
				//if so, the value in the array is set to 1. If not, it is set to 0.
				if(Dataset.compare(data[lineTracker].Y, finalValues)){
					correct[lineTracker] = 1;
				}else{
					correct[lineTracker] = 0;
				}
				
				//point to next line
				lineTracker++;	
			}
		
			//Close resources
			br.close();
			fr.close();
		}
			//testing values
			for(int i = 0; i < correct.length; i++){
				correct[i] = (int)Math.round(Math.random());
			}
			return correct;
	}
	
	//Option 5: runs the network on testing data showing images and labels after selecting either 1 or 2
	//Values is either the training or testing dataset
	public static void runNetwork(String decision) throws IOException{
		System.out.println("Loading images . . .");
		//use decision to determin which dataset to use
		File file;
		int datasetSize;
		if(decision.equals("test")){
			file = new File("mnist_test.csv");
			datasetSize = 10000;
		}
		else{
			file = new File("mnist_train.csv");
			datasetSize = 60000;
		}
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		Scanner scanner = new Scanner(System.in);
		
		//Create dataset from file
		Dataset[] data = new Dataset[datasetSize];
		String valuesString;
		int lineTracker = 0;
		while((valuesString = br.readLine()) != null){
			//creating input (X)
			String xString = valuesString.substring(2);
			//creating list of doubles
			double[] xValues = new double[784];
			//converting string of values to array of values
			int strIterate = 0;
			int arrIterate = 0;
			while(strIterate < xString.length()){
				String currentNum = "";
				//create a string of the next number based on where the next "," is
				while(Character.isDigit(xString.charAt(strIterate))){
					currentNum += xString.charAt(strIterate);
					strIterate++;
					//check if at end of line
					if(strIterate >= xString.length()){break;}
				}
				double currentValue = Double.parseDouble(currentNum);
				xValues[arrIterate] = currentValue;
				//skip comma
				strIterate++;
				//move to next array slot
				arrIterate++;
			}
			//converting arry of values to Matrix of values
			Matrix X = new Matrix(xValues);
			X = X.transposeMatrix();
			
			//creating expected output (Y)
			//grabbing the value as a char
			int outputVal = Integer.valueOf(valuesString.substring(0,1));
			double[] yVals = new double[10];
			Arrays.fill(yVals, 0.0);
			yVals[outputVal] = 1.0;
			Matrix Y = new Matrix(yVals);
			Y = Y.transposeMatrix();
			
			data[lineTracker] = new Dataset(X, Y);
			lineTracker++;
		}
		
		//testing the network one at a time
		int i;
		lineTracker = 0;
		String acknowledgement = "1";
		int[] correct = trackAccuracy(data);
		while(acknowledgement.equals("1") && lineTracker < data.length){	
			//display graphics
			i = 0;
			//Determine outcome of test
			String outcome;
			if(correct[lineTracker] == 0){outcome = "incorrect";}
			else{outcome = "correct";}
			//Loops until the user exits (enters 1) or the end of the testing data is reached
			//Top text
			Matrix hiddenValues = data[lineTracker].X.forwardPropagate(weights1, biases1);
			Matrix fP = hiddenValues.forwardPropagate(weights2, biases2);
			System.out.println("Testing Case #" + lineTracker + ": Correct Classification = " + Dataset.getIntY(data[lineTracker].Y) + " Network Output = " + Dataset.getIntY(fP) + "  " + outcome + ".");
			//Displays ASCII art of number in 28x28 grid
			for(int j = 0; j < 28; j++){
				for(int k = 0; k < 28; k++){
					// int asciiVal = (int)Math.round(data[lineTracker].X.getValue(i,0));
					// System.out.print(Character.forDigit(asciiVal, 10));
					if(data[lineTracker].X.getValue(i,0) == 0){System.out.print("@");}
					else{System.out.print("#");}
					i++;
				}
				System.out.println();
			}
			
			//Bottom text
			System.out.println("\nEnter 1 to continue. All other values return to main menu.\n");
			
			//get user permission to continue to next line
			lineTracker++;
			acknowledgement = scanner.next();
		}
		
		//close resources
		fr.close();
		br.close();
	}
	
	//Option 6: displays misclassified images after selecting either 1 or 2
	public static void displayMisclassifiedImages(String decision) throws IOException{
		System.out.println("Loading images . . .");
		//use decision to determin which dataset to use
		File file;
		int datasetSize;
		if(decision.equals("test")){
			file = new File("mnist_test.csv");
			datasetSize = 10000;
		}
		else{
			file = new File("mnist_train.csv");
			datasetSize = 60000;
		}
		FileReader fr = new FileReader(file);
		BufferedReader br = new BufferedReader(fr);
		Scanner scanner = new Scanner(System.in);
		
		//Create dataset from file
		Dataset[] data = new Dataset[datasetSize];
		String valuesString;
		int lineTracker = 0;
		while((valuesString = br.readLine()) != null){
			//creating input (X)
			String xString = valuesString.substring(2);
			//creating list of doubles
			double[] xValues = new double[784];
			//converting string of values to array of values
			int strIterate = 0;
			int arrIterate = 0;
			while(strIterate < xString.length()){
				String currentNum = "";
				//create a string of the next number based on where the next "," is
				while(Character.isDigit(xString.charAt(strIterate))){
					currentNum += xString.charAt(strIterate);
					strIterate++;
					//check if at end of line
					if(strIterate >= xString.length()){break;}
				}
				double currentValue = Double.parseDouble(currentNum);
				xValues[arrIterate] = currentValue;
				//skip comma
				strIterate++;
				//move to next array slot
				arrIterate++;
			}
			//converting arry of values to Matrix of values
			Matrix X = new Matrix(xValues);
			X = X.transposeMatrix();
			
			//creating expected output (Y)
			//grabbing the value as a char
			int outputVal = Integer.valueOf(valuesString.substring(0,1));
			double[] yVals = new double[10];
			Arrays.fill(yVals, 0.0);
			yVals[outputVal] = 1.0;
			Matrix Y = new Matrix(yVals);
			Y = Y.transposeMatrix();
			
			data[lineTracker] = new Dataset(X, Y);
			lineTracker++;
		}
		
		//testing the network one at a time
		int i;
		lineTracker = 0;
		String acknowledgement = "1";
		int[] correct = trackAccuracy(data);
		while(acknowledgement.equals("1") && lineTracker < data.length){	
			//display graphics
			i = 0;
			//Determine outcome of test
			String outcome;
			if(correct[lineTracker] == 0){
				outcome = "incorrect";
			}
			else{
				outcome = "correct";
			}
			//Loops until the user exits (enters 1) or the end of the testing data is reached
			if(outcome.equals("incorrect")){
				//Top text
				Matrix hiddenValues = data[lineTracker].X.forwardPropagate(weights1, biases1);
				Matrix fP = hiddenValues.forwardPropagate(weights2, biases2);
				System.out.println("Testing Case #" + lineTracker + ": Correct Classification = " + Dataset.getIntY(data[lineTracker].Y) + " Network Output = " + Dataset.getIntY(fP) + "  " + outcome + ".");
				//Displays ASCII art of number in 28x28 grid
				for(int j = 0; j < 28; j++){
					for(int k = 0; k < 28; k++){
						// int asciiVal = (int)Math.round(data[lineTracker].X.getValue(i,0));
						// System.out.print(Character.forDigit(asciiVal, 10));
						if(data[lineTracker].X.getValue(i,0) == 0){System.out.print("@");}
						else{System.out.print("#");}
						i++;
					}
					System.out.println();
				}
				//Bottom text
				System.out.println("\nEnter 1 to continue. All other values return to main menu.\n");
				acknowledgement = scanner.next();
			}
			
			//get user permission to continue to next line
			lineTracker++;
		}
		
		//close resources
		fr.close();
		br.close();
	}
	
	//Option 7: saves the current set of W/B to a file after selecting either 1 or 2
	public static void saveState() throws IOException{
		System.out.println("Saving Neural Network to file . . .");
		//layer 1
		File weightFile1 = new File("mnist_weights1.csv");
		FileWriter fwweight1 = new FileWriter(weightFile1);
		BufferedWriter bwweight1 = new BufferedWriter(fwweight1);
		
		File biasFile1 = new File("mnist_biases1.csv");
		FileWriter fwbias1 = new FileWriter(biasFile1);
		BufferedWriter bwbias1 = new BufferedWriter(fwbias1);
		//Save weights
		//Parse each line
		for(int i = 0; i < CLI.weights1.getMatrix().length; i++){
			//Loop putting values onto file as Strings with commas between
			for(int j = 0; j < CLI.weights1.getArray(0).length; j++){
				bwweight1.write(Double.toString(CLI.weights1.getValue(i,j)));
				if(!(j == CLI.weights1.getArray(0).length - 1)){bwweight1.write(",");}
			}
			//At end of each row move to next line of file
			bwweight1.newLine();
		}
		//Save biases
		//Parse each line
		for(int i = 0; i < CLI.biases1.getMatrix().length; i++){
			//Loop putting values onto file as Strings with commas between
			for(int j = 0; j < CLI.biases1.getArray(0).length; j++){
				bwbias1.write(Double.toString(CLI.biases1.getValue(i,j)));
				if(!(j == CLI.biases1.getArray(0).length - 1)){bwbias1.write(",");}
			}
			//At end of each row move to next line of file
			bwbias1.newLine();
		}
		
		//layer 2
		File weightFile2 = new File("mnist_weights2.csv");
		FileWriter fwweight2 = new FileWriter(weightFile2);
		BufferedWriter bwweight2 = new BufferedWriter(fwweight2);
		
		File biasFile2 = new File("mnist_biases2.csv");
		FileWriter fwbias2 = new FileWriter(biasFile2);
		BufferedWriter bwbias2 = new BufferedWriter(fwbias2);
		//Save weights
		//Parse each line
		for(int i = 0; i < CLI.weights2.getMatrix().length; i++){
			//Loop putting values onto file as Strings with commas between
			for(int j = 0; j < CLI.weights2.getArray(0).length; j++){
				bwweight2.write(Double.toString(CLI.weights2.getValue(i,j)));
				if(!(j == CLI.weights2.getArray(0).length - 1)){bwweight2.write(",");}
			}
			//At end of each row move to next line of file
			bwweight2.newLine();
		}
		//Save biases
		//Parse each line
		for(int i = 0; i < CLI.biases2.getMatrix().length; i++){
			//Loop putting values onto file as Strings with commas between
			for(int j = 0; j < CLI.biases2.getArray(0).length; j++){
				bwbias2.write(Double.toString(CLI.biases2.getValue(i,j)));
				if(!(j == CLI.biases2.getArray(0).length - 1)){bwbias2.write(",");}
			}
			//At end of each row move to next line of file
			bwbias2.newLine();
		}
		
		//close objects
		bwweight1.close();
		fwweight1.close();
		bwbias1.close();
		fwbias1.close();
		bwweight2.close();
		fwweight2.close();
		bwbias2.close();
		fwbias2.close();
		
		System.out.println("Neural Network successfully saved to file. It is now safe to exit.");

	}
	
	//Option 8: exits the program
	public static void exitNetwork(){
		System.out.println("Understood: exiting. Have a good day!");
		System.exit(0);
	}
}


class Main{
	public static void main(String[] args) throws IOException{
		//Testing functions
			/*
			Finished:
			--Matrix class
			--Dataset class
			--saveState
			--exitNetwork
			--loadNetwork
			--trainNetwork
			--trackAccuracy
			--getIntY
			--runNetwork
			--displayAccuracy
			--main
			--displayMisclassifiedImages
			
			In Progress:
			--compare: get professor's feedback about approach
			*/
		
		Scanner scanner = new Scanner(System.in);
		while(true){
			//Starting screen upon running program - first selection
			System.out.println("\n\nWelcome to the Neural Network. Type a number to select an option:\n" +
				"0: Exit\n" +
				"1: Train the network\n" +
				"2: Load a pre-trained network\n"
			);
			System.out.print(">> ");
			
			//decision string will save whether we are working with
			//train or test and will submit that to the desired function
			String decision = "";
			int option1 = -1;
			try{
				option1 = Integer.parseInt(scanner.next());
			} catch(Exception e){
				System.out.println("Unrecognized input.");
				continue;
			}
			//Do something with the data
			switch (option1){
				case 0:
					CLI.exitNetwork();
					break;
				case 1:
					System.out.println("Your selection: " + option1 + "\n");
					decision = "train";
					CLI.trainNetwork();
					break;
				case 2:
					System.out.println("Your selection: " + option1 + "\n");
					decision = "test";
					CLI.loadNetwork();
					break;
				default:
					System.out.println("Unrecognized input.");
					continue;
			}
			
			//Second selection
			System.out.println("\n\nType a number to select an option:\n" +
				"0: Exit\n" +
				"3: Display network accuracy on training data\n" +
				"4: Display network accuracy on testing data\n" +
				"5: Run network on testing data showing images and labels\n" +
				"6: Display misclassified testing images\n" +
				"7: Save network state\n"
			);
			System.out.print(">> ");
			
			int option2 = -1;
			try{
				option2 = Integer.parseInt(scanner.next());
			} catch(Exception e){
				System.out.println("Unrecognized input.");
				continue;
			}
			//Do something with the data
			switch (option2){
				case 0:
					CLI.exitNetwork();
					break;
				case 3:
					System.out.println("Your selection: " + option2 + "\n");
					CLI.displayAccuracy("train");
					break;
				case 4:
					System.out.println("Your selection: " + option2 + "\n");
					CLI.displayAccuracy("test");
					break;
				case 5:
					System.out.println("Your selection: " + option2 + "\n");
					CLI.runNetwork(decision);
					break;
				case 6:
					System.out.println("Your selection: " + option2 + "\n");
					CLI.displayMisclassifiedImages(decision);
					break;
				case 7:
					System.out.println("Your selection: " + option2 + "\n");
					CLI.saveState();
					break;
				default:
					System.out.println("Unrecognized input.");
					break;
			}
		}
	}
}