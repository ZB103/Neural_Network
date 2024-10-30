/*
Assignment 2-1 for CSC 475 - small neural network implementation
Zoe Baker
10/10/24
*/
import java.util.*;

//Matrix class that handles both 1D and 2D Arrays with arbitrary parameters
class Matrix{
	public static int learningRate = 10;
	public static int minibatchSize = 2;
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
	public Matrix adjustWeightBias(Matrix case1Gradient, Matrix case2Gradient){
		Matrix sum = case1Gradient.addMatrices(case2Gradient);
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
		System.out.println();
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


class Main{
	public static void main(String[] args){
		//Creating information
		//X's (input values) and Y's (expected outputs)
		Matrix X1 = new Matrix(new double[]{0.0,1.0,0.0,1.0});
		X1 = X1.transposeMatrix();
		Matrix Y1 = new Matrix(new double[]{0.0,1.0});
		Y1 = Y1.transposeMatrix();
		Matrix X2 = new Matrix(new double[]{1.0,0.0,1.0,0.0});
		X2 = X2.transposeMatrix();
		Matrix Y2 = new Matrix(new double[]{1.0,0.0});
		Y2 = Y2.transposeMatrix();
		Matrix X3 = new Matrix(new double[]{0.0,0.0,1.0,1.0});
		X3 = X3.transposeMatrix();
		Matrix Y3 = new Matrix(new double[]{0.0,1.0});
		Y3 = Y3.transposeMatrix();
		Matrix X4 = new Matrix(new double[]{1.0,1.0,0.0,0.0});
		X4 = X4.transposeMatrix();
		Matrix Y4 = new Matrix(new double[]{1.0,0.0});
		Y4 = Y4.transposeMatrix();
		
		//The weights connecting L0 and L1
		Matrix W1 = new Matrix(
			new double[]{-0.21,0.72,-0.25,1.0},
			new double[]{-0.94,-0.41,-0.47,0.63},
			new double[]{0.15,0.55,-0.49,-0.75}
		);
		Matrix B1 = new Matrix(
			new double[]{0.1},
			new double[]{-0.36},
			new double[]{-0.31}
		);
		//The weights connecting L1 and L2
		Matrix W2 = new Matrix(
			new double[]{0.76,.48,-0.73},
			new double[]{0.34,0.89,-0.23}
		);
		Matrix B2 = new Matrix(
			new double[]{0.16},
			new double[]{-0.46}
		);
	
		//Start learning
		for(int i = 0; i < 6; i++){	//epoch
			//minibatch 1: data set 1 and 2
				//data set 1: X1 and Y1
					//forward
					Matrix hiddenValues1 = X1.forwardPropagate(W1, B1);
					Matrix finalValues1 = hiddenValues1.forwardPropagate(W2, B2);
					
					System.out.println("hiddenValues");
					hiddenValues1.printMatrix();
					System.out.println("finalValues");
					finalValues1.printMatrix();
					//back
					Matrix outputBiasGrad1 = finalValues1.getOutputBiasGrad(Y1);
					Matrix outputWeightGrad1 = hiddenValues1.getWeightGrad(outputBiasGrad1);
					Matrix hiddenBiasGrad1 = hiddenValues1.getHiddenBiasGrad(W2, outputBiasGrad1);
					Matrix hiddenWeightGrad1 = X1.getWeightGrad(hiddenBiasGrad1);
					
					System.out.println("outputBiasGrad");
					outputBiasGrad1.printMatrix();
					System.out.println("outputWeightGrad");
					outputWeightGrad1.printMatrix();
					System.out.println("hiddenBiasGrad");
					hiddenBiasGrad1.printMatrix();
					System.out.println("hiddenWeightGrad");
					hiddenWeightGrad1.printMatrix();
				//data set 2: X2 and Y2
					//forward
					Matrix hiddenValues2 = X2.forwardPropagate(W1, B1);
					Matrix finalValues2 = hiddenValues2.forwardPropagate(W2, B2);
					
					System.out.println("hiddenValues");
					hiddenValues2.printMatrix();
					System.out.println("finalValues");
					finalValues2.printMatrix();
					//back
					Matrix outputBiasGrad2 = finalValues2.getOutputBiasGrad(Y2);
					Matrix outputWeightGrad2 = hiddenValues2.getWeightGrad(outputBiasGrad2);
					Matrix hiddenBiasGrad2 = hiddenValues2.getHiddenBiasGrad(W2, outputBiasGrad2);
					Matrix hiddenWeightGrad2 = X2.getWeightGrad(hiddenBiasGrad2);
					
					System.out.println("outputBiasGrad");
					outputBiasGrad2.printMatrix();
					System.out.println("outputWeightGrad");
					outputWeightGrad2.printMatrix();
					System.out.println("hiddenBiasGrad");
					hiddenBiasGrad2.printMatrix();
					System.out.println("hiddenWeightGrad");
					hiddenWeightGrad2.printMatrix();
					
				//adjustments
				B2 = B2.adjustWeightBias(outputBiasGrad1, outputBiasGrad2);
				W2 = W2.adjustWeightBias(outputWeightGrad1, outputWeightGrad2);
				B1 = B1.adjustWeightBias(hiddenBiasGrad1, hiddenBiasGrad2);
				W1 = W1.adjustWeightBias(hiddenWeightGrad1, hiddenWeightGrad2);
				
				System.out.println("B2");
				B2.printMatrix();
				System.out.println("W2");
				W2.printMatrix();
				System.out.println("B1");
				B1.printMatrix();
				System.out.println("W1");
				W1.printMatrix();
			
			
			//minibatch 1: data set 3 and 4
				//data set 1: X3 and Y3
					//forward
					Matrix hiddenValues3 = X3.forwardPropagate(W1, B1);
					Matrix finalValues3 = hiddenValues3.forwardPropagate(W2, B2);
					System.out.println("hiddenValues");
					hiddenValues3.printMatrix();
					System.out.println("finalValues");
					finalValues3.printMatrix();
					//back
					Matrix outputBiasGrad3 = finalValues3.getOutputBiasGrad(Y3);
					System.out.println("outputBiasGrad");
					outputBiasGrad3.printMatrix();
					Matrix outputWeightGrad3 = hiddenValues3.getWeightGrad(outputBiasGrad3);
					System.out.println("outputWeightGrad");
					outputWeightGrad3.printMatrix();
					Matrix hiddenBiasGrad3 = hiddenValues3.getHiddenBiasGrad(W2, outputBiasGrad3);
					System.out.println("hiddenBiasGrad");
					hiddenBiasGrad3.printMatrix();
					Matrix hiddenWeightGrad3 = X3.getWeightGrad(hiddenBiasGrad3);
					System.out.println("hiddenWeightGrad");
					hiddenWeightGrad3.printMatrix();
				//data set 2: X2 and Y2
					//forward
					Matrix hiddenValues4 = X4.forwardPropagate(W1, B1);
					Matrix finalValues4 = hiddenValues4.forwardPropagate(W2, B2);
					System.out.println("hiddenValues");
					hiddenValues4.printMatrix();
					System.out.println("finalValues");
					finalValues4.printMatrix();
					//back
					Matrix outputBiasGrad4 = finalValues4.getOutputBiasGrad(Y4);
					System.out.println("outputBiasGrad");
					outputBiasGrad4.printMatrix();
					Matrix outputWeightGrad4 = hiddenValues4.getWeightGrad(outputBiasGrad4);
					System.out.println("outputWeightGrad");
					outputWeightGrad4.printMatrix();
					Matrix hiddenBiasGrad4 = hiddenValues4.getHiddenBiasGrad(W2, outputBiasGrad4);
					System.out.println("hiddenBiasGrad");
					hiddenBiasGrad4.printMatrix();
					Matrix hiddenWeightGrad4 = X4.getWeightGrad(hiddenBiasGrad4);
					System.out.println("hiddenWeightGrad");
					hiddenWeightGrad4.printMatrix();
					
				//adjustments
				B2 = B2.adjustWeightBias(outputBiasGrad3, outputBiasGrad4);
				System.out.println("B2");
				B2.printMatrix();
				W2 = W2.adjustWeightBias(outputWeightGrad3, outputWeightGrad4);
				System.out.println("W2");
				W2.printMatrix();
				B1 = B1.adjustWeightBias(hiddenBiasGrad3, hiddenBiasGrad4);
				System.out.println("B1");
				B1.printMatrix();
				W1 = W1.adjustWeightBias(hiddenWeightGrad3, hiddenWeightGrad4);
				System.out.println("W1");
				W1.printMatrix();
		}
	}
}