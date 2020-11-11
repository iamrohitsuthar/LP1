public class State {
	int arr[];
	State paraent;
	int h = 0;

	State(State s1) {
		this.arr = new int[9];
		for (int i = 0; i < s1.arr.length; i++) {
			this.arr[i] = s1.arr[i];
		}

	}

	State() {
		arr = new int[9];
	}
}
