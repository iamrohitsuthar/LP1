import java.util.ArrayList;
import java.util.Scanner;

public class HillClimbingAlgo {
	State gstate, cstate, sstate;
	Scanner sc = new Scanner(System.in);
	ArrayList<State> ngb = new ArrayList<State>();

	HillClimbingAlgo() {
		gstate = new State();
		cstate = new State();
		sstate = new State();
	}

	void display(State s) {
		int k = 0;
		for (int j = 0; j < 3; j++) {
			for (int i = 0; i < 3; i++) {
				System.out.print(s.arr[k] + " ");
				k++;
			}
			System.out.println();
		}

	}

	void input() {

		System.out.println("Enter the start state");
		for (int i = 0; i < 9; i++) {
			sstate.arr[i] = sc.nextInt();
		}

		System.out.println("Enter the goal state");
		for (int i = 0; i < 9; i++) {
			gstate.arr[i] = sc.nextInt();
		}

	}

	int h(State s) {
		int hvalue = 0;
		for (int i = 0; i < 9; i++) {
			if (s.arr[i] != gstate.arr[i]) {
				hvalue++;
			}
		}

		return hvalue;
	}

	int blpos(State s) {
		for (int j = 0; j < 9; j++) {
			if (s.arr[j] == 0) {
				return j;
			}
		}
		return 0;
	}

	void Movegen(State s) {
		int p = blpos(s);
		ngb.clear();
		if (p % 3 != 0) {
			State n1 = new State(s);
			n1.arr[p] = n1.arr[p - 1];
			n1.arr[p - 1] = 0;
			n1.h = h(n1);
			ngb.add(n1);
		}

		if (p < 6) {
			State n1 = new State(s);
			n1.arr[p] = n1.arr[p + 3];
			n1.arr[p + 3] = 0;
			n1.h = h(n1);
			ngb.add(n1);
		}
		if (p > 2 && p < 9) {
			State n1 = new State(s);
			n1.arr[p] = n1.arr[p - 3];
			n1.arr[p - 3] = 0;
			n1.h = h(n1);
			ngb.add(n1);
		}
		if (p % 3 != 2) {
			State n1 = new State(s);
			n1.arr[p] = n1.arr[p + 1];
			n1.arr[p + 1] = 0;
			n1.h = h(n1);
			ngb.add(n1);
		}

	}

	int lowestscore() {
		int i = 0, min = 999;
		for (int j = 0; j < ngb.size(); j++) {
			if (min > ngb.get(j).h) {
				min = ngb.get(j).h;
				i = j;
			}
		}

		return i;
	}

	State hillclimbing() {
		int low = 0, done = 0;
		State n, nn;
		sstate.h = h(sstate);
		sstate.paraent = null;
		n = sstate;
		Movegen(n);
		low = lowestscore();
		nn = ngb.get(low);
		display(n);
		System.out.println();
		while (nn.h < n.h) {
			display(nn);
			System.out.println();
			nn.paraent = n;
			n = nn;

			Movegen(n);
			low = lowestscore();
			nn = ngb.get(low);
		}
		return nn;

	}

	public static void main(String[] args) {
		HillClimbingAlgo ob = new HillClimbingAlgo();
		ob.input();
		System.out.println("Intermediate States");
		ob.hillclimbing();
	}
}
