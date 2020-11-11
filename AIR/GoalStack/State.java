public class State {
    int on[][];
    int hold[];
    int clear[];
    int ontable[];
    Boolean arm_empty;

    public State(int n,String state){
        on=new int[n][n];
        hold=new int[n];
        clear=new int[n];
        ontable=new int[n];
        arm_empty=Boolean.TRUE;
        setState(state);

    }

    public void setState(String state){
        String subgoals[]=state.split("['^']+");
        for(String sub:subgoals){
            String elems[]=sub.split("[() ]+");
            if(elems[1].equals("on")){
                on[elems[2].charAt(0)-97][elems[3].charAt(0)-97]=1;
            }
            else if(elems[1].equals("ontable")){
                ontable[elems[2].charAt(0)-97]=1;
            }
            else if(elems[1].equals("clear")){
                clear[elems[2].charAt(0)-97]=1;
            }
            else if(elems[1].equals("hold")){
                hold[elems[2].charAt(0)-97]=1;
            }
            else if(sub.contains("AE")){
                arm_empty=Boolean.TRUE;
            }
        }
    }

	//satisfy compares current state with goal state ,if same return true
    public Boolean satisfy(String goal){
        String subgoals[]=goal.split("['^']+");
        for(String sub:subgoals){
            String elems[]=sub.split("[() ]+");
            if(elems[1].equals("on") && on[elems[2].charAt(0)-97][elems[3].charAt(0)-97]==1){
                continue;
            }
            else if(elems[1].equals("ontable") && ontable[elems[2].charAt(0)-97]==1){
                continue;
            }
            else if(elems[1].equals("clear") && clear[elems[2].charAt(0)-97]==1){
                continue;
            }
            else if(elems[1].equals("hold") && hold[elems[2].charAt(0)-97]==1){
                continue;
            }
            else if(elems[1].equals("AE") && arm_empty==Boolean.TRUE){
                continue;
            }
            else{
                return Boolean.FALSE;
            }
        }
        return Boolean.TRUE;
    }

    public void prformAction(String action){
        String elems[]=action.split("[() ]+");
        if(elems[1].equals("putdown")){
            ontable[elems[2].charAt(0)-97]=1;
            clear[elems[2].charAt(0)-97]=1;
            hold[elems[2].charAt(0)-97]=0;
            arm_empty=Boolean.TRUE;
        }
        else if(elems[1].equals("pick")){
            ontable[elems[2].charAt(0)-97]=0;
            clear[elems[2].charAt(0)-97]=0;
            hold[elems[2].charAt(0)-97]=1;
            arm_empty=Boolean.FALSE;
        }
        else if(elems[1].equals("unstack")){
            on[elems[2].charAt(0)-97][elems[3].charAt(0)-97]=0;
            clear[elems[2].charAt(0)-97]=0;
            clear[elems[3].charAt(0)-97]=1;
            hold[elems[2].charAt(0)-97]=1;
            arm_empty=Boolean.FALSE;
        }
        else if(elems[1].equals("stack")){
            on[elems[2].charAt(0)-97][elems[3].charAt(0)-97]=1;
            clear[elems[2].charAt(0)-97]=1;
            clear[elems[3].charAt(0)-97]=0;
            hold[elems[2].charAt(0)-97]=0;
            arm_empty=Boolean.TRUE;
        }
    }

    public int checkTop(char c){
        int len=hold.length;
        for(int i=0;i<len;i++){
            if(on[i][c-97]==1){
                return i;
            }
        }
        return -1;
    }

}