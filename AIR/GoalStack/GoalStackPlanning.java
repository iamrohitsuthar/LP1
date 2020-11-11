import java.util.ArrayList;
import java.util.Scanner;
import java.util.Stack;

public class GoalStackPlanning {
    State curr,goal;
    String goal_state;
    ArrayList<String> steps;
    Stack<String> st;
    int n;

    public GoalStackPlanning(int n, String start,String Goal){
        this.n=n;
        this.goal_state=new String(Goal);
        curr= new State(n,start);
        this.goal=new State(n,Goal);
        steps=new ArrayList<String>();
        st=new Stack<String>();
        st.push(goal_state);
    }
    
    public void plan(){
        String s[]=goal_state.split("['^']");
        for(int i=s.length-1;i>=0;i--){
            st.push(s[i]);
        }
        while(!st.empty()){

            String sub=st.pop();
            if(sub.contains("^")){
                if(sub.equals(goal_state) && !curr.satisfy(sub)){
                    st.push(goal_state);
                }
                if(!curr.satisfy(sub)){
                    String goals[]=sub.split("['^']");
                    for(int i=goals.length-1;i>=0;i--){
                        st.push(goals[i]);
                    }
                }
            }

            else if(sub.contains("on ") && !curr.satisfy(sub)){
                String elements[]=sub.split("[() ]+");
                st.push("(stack "+elements[2].charAt(0)+" " + elements[3]+")");
                st.push("(clear "+elements[2].charAt(0)+")^(clear "+elements[3]+")^(AE)");
            }
            else if(sub.contains("ontable") && !curr.satisfy(sub)){
                String elements[]=sub.split("[() ]+");
                st.push("(putdown "+elements[2].charAt(0)+")");
                st.push("(hold "+elements[2].charAt(0)+")");
            }
            else if(sub.contains("clear") && !curr.satisfy(sub)){
                String elements[]=sub.split("[() ]+");
                if(curr.hold[elements[2].charAt(0)-97]==1){
                    st.push("(putdown "+elements[2].charAt(0)+")");
                    st.push("(hold "+elements[2].charAt(0)+")");
                }
                else{
                    int temp=curr.checkTop(elements[2].charAt(0));
                    if(temp!=-1){
                        st.push("(unstack "+Character.toString((char)(temp+97))+" "+elements[2].charAt(0)+")");
                        st.push("(on "+Character.toString((char)(temp+97))+" "+elements[2].charAt(0)+")^(clear "+Character.toString((char)(temp+97))+")^(AE)");
                    }
                }
            }
            else if(sub.contains("hold") && !curr.satisfy(sub)){
                String elements[]=sub.split("[() ]+");
                if(curr.ontable[elements[2].charAt(0)-97]==1){
                    st.push("(pick "+elements[2].charAt(0)+")");
                    st.push("(ontable "+elements[2].charAt(0)+")^(clear "+elements[2].charAt(0)+")^(AE)");
                }
                else{
                    int temp=curr.checkTop(elements[2].charAt(0));
                    if(temp!=-1){
                        st.push("(unstack "+Character.toString((char)(temp+97))+" "+elements[2].charAt(0)+")");
                        st.push("(on "+Character.toString((char)(temp+97))+" "+elements[2].charAt(0)+")^(clear "+Character.toString((char)(temp+97))+")^(AE)");
                    }
                }
            }
            else if(sub.contains("AE") && !curr.satisfy(sub)){
                for(int i=0;i<n;i++){
                    if(curr.hold[i]==1){
                        st.push("(putdown "+Character.toString((char)(i+97))+")");
                        st.push("(hold "+Character.toString((char)(i+97))+")");
                    }
                }
            }
            else if(sub.contains("pick") || sub.contains("putdown") || sub.contains("stack") || sub.contains("unstack")){
                curr.prformAction(sub);
                steps.add(sub);
            }

        }
    }

    public void printSteps(){
        System.out.println();
        System.out.println("Steps taken:");
        for(String step:steps){
            System.out.println(step);
        }
    }

    public static void main(String[] args) {
        int n;
        String start,goal;
        Scanner sc=new Scanner(System.in);
        System.out.println("Enter no of blocks");
        n=sc.nextInt();
        sc.nextLine();
        System.out.println("Enter Start state: ");
        start=sc.nextLine();
        System.out.println("Enter goal state: ");
        goal=sc.nextLine();

        GoalStackPlanning obj=new GoalStackPlanning(n,start,goal);
        obj.plan();
        obj.printSteps();
    }
}