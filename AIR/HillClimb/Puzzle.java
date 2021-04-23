import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.*;

public class Puzzle {
	
	public int dimension = 3;
	
	// Moves
	int[] row = { 1, 0, -1, 0 };
	int[] col = { 0, -1, 0, 1 };

	PriorityQueue<Node> pq = new PriorityQueue<Node>(100, (a, b) -> a.cost - b.cost);
	
	public int calculateCost(int[][] initial, int[][] goal) {
		int count = 0;
		int n = initial.length;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (initial[i][j] != 0 && initial[i][j] != goal[i][j]) {
					count++;
				}
			}
		}
		return count;
	}
	
	public void printMatrix(int[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix.length; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	public boolean isSafe(int x, int y) {
		return (x >= 0 && x < dimension && y >= 0 && y < dimension);
	}
	
	public boolean isSolvable(int[][] matrix) {
		int count = 0;
		List<Integer> array = new ArrayList<Integer>();
		
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix.length; j++) {
				array.add(matrix[i][j]);
			}
		}
		
		Integer[] anotherArray = new Integer[array.size()];
		array.toArray(anotherArray);
		
		for (int i = 0; i < anotherArray.length - 1; i++) {
			for (int j = i + 1; j < anotherArray.length; j++) {
				if (anotherArray[i] != 0 && anotherArray[j] != 0 && anotherArray[i] > anotherArray[j]) {
					count++;
				}
			}
		}
		
		return count % 2 == 0;
	}

	public int generateChilds(Node node, int[][] goal) {
		pq.clear();
		for (int i = 0; i < 4; i++) {
            if (isSafe(node.x + row[i], node.y + col[i])) {
            	Node child = new Node(node.matrix, node.x, node.y, node.x + row[i], node.y + col[i]);
            	child.cost = calculateCost(child.matrix, goal);
            	if(child.cost == 0) { 
            		printMatrix(child.matrix);
            		return 1;
            	}
            	pq.add(child);
            }
	    }
	    return 0;
	}

	public void solvePuzzle(int[][] initial, int[][] goal, int x, int y) {
		Node currentState, newState;
		Node root = new Node(initial, x, y, x, y);
		root.cost = calculateCost(initial, goal);
		printMatrix(root.matrix);
		currentState = root;

		if(root.cost == 0) return;

		if(generateChilds(currentState, goal) == 1) return;
		newState = pq.poll();
		printMatrix(newState.matrix);

		while (newState.cost < currentState.cost) {
			currentState = newState;
			if(generateChilds(currentState, goal) == 1) break;
			newState = pq.poll();
	        printMatrix(newState.matrix);
		}
	}

	public int[] findTilePosition(int initial[][]) {
		int res[] = new int[2];
		for(int i = 0 ; i < initial.length ; i++) {
			for(int j = 0 ; j < initial[0].length ; j++) {
				if(initial[i][j] == 0) {
					res[0] = i;
					res[1] = j;
				}
			}
		}
		return res;
	}
	
	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		int initial[][] = new int[3][3];
		int goal[][] = new int[3][3];

		System.out.println("Enter Initial Matrix: ");
		for(int i = 0 ; i < initial.length ; i++) {
			for(int j = 0 ; j < initial[0].length ; j++) {
				initial[i][j] = scanner.nextInt();
			}
		}	

		System.out.println("Enter Goal Matrix: ");
		for(int i = 0 ; i < initial.length ; i++) {
			for(int j = 0 ; j < initial[0].length ; j++) {
				goal[i][j] = scanner.nextInt();
			}
		}
		System.out.println();
	
		Puzzle puzzle = new Puzzle();

		int res[] = puzzle.findTilePosition(initial);

		if (puzzle.isSolvable(initial)) {
			puzzle.solvePuzzle(initial, goal, res[0], res[1]);
		} 
		else {
			System.out.println("Puzzle is not solvable");
		}
	}

}

class Node {

	public int[][] matrix;
	public int x, y; //blank title coordinates
	public int cost; // misplaced tiles - h value
	
	public Node(int[][] matrix, int x, int y, int newX, int newY) {
		this.matrix = new int[matrix.length][];
		for (int i = 0; i < matrix.length; i++) {
			this.matrix[i] = matrix[i].clone();
		}
		
		int temp = this.matrix[x][y];
		this.matrix[x][y] = this.matrix[newX][newY];
		this.matrix[newX][newY] = temp;

		this.cost = Integer.MAX_VALUE;
		this.x = newX;
		this.y = newY;
	}
}