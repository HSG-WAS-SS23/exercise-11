package tools;

import java.util.*;
import java.util.logging.*;
import cartago.Artifact;
import cartago.OPERATION;
import cartago.OpFeedbackParam;

public class QLearner extends Artifact {

  private Lab lab; // the lab environment that will be learnt 
  private int stateCount; // the number of possible states in the lab environment
  private int actionCount; // the number of possible actions in the lab environment
  private HashMap<Integer, double[][]> qTables; // a map for storing the qTables computed for different goals

  private static final Logger LOGGER = Logger.getLogger(QLearner.class.getName());

  public void init(String environmentURL) {

    // the URL of the W3C Thing Description of the lab Thing
    this.lab = new Lab(environmentURL);

    this.stateCount = this.lab.getStateCount();
    LOGGER.info("Initialized with a state space of n="+ stateCount);

    this.actionCount = this.lab.getActionCount();
    LOGGER.info("Initialized with an action space of m="+ actionCount);

    qTables = new HashMap<>();

  }

/**
* Computes a Q matrix for the state space and action space of the lab, and against
* a goal description. For example, the goal description can be of the form [z1level, z2Level],
* where z1Level is the desired value of the light level in Zone 1 of the lab,
* and z2Level is the desired value of the light level in Zone 2 of the lab.
* For exercise 11, the possible goal descriptions are:
* [0,0], [0,1], [0,2], [0,3], 
* [1,0], [1,1], [1,2], [1,3], 
* [2,0], [2,1], [2,2], [2,3], 
* [3,0], [3,1], [3,2], [3,3].
*
*<p>
* HINT: Use the methods of {@link LearningEnvironment} (implemented in {@link Lab})
* to interact with the learning environment (here, the lab), e.g., to retrieve the
* applicable actions, perform an action at the lab during learning etc.
*</p>
* @param  goalDescription  the desired goal against the which the Q matrix is calculated (e.g., [2,3])
* @param  episodesObj the number of episodes used for calculating the Q matrix
* @param  alphaObj the learning rate with range [0,1].
* @param  gammaObj the discount factor [0,1]
* @param epsilonObj the exploration probability [0,1]
* @param rewardObj the reward assigned when reaching the goal state
**/
  @OPERATION
  public void calculateQ(Object[] goalDescription , Object episodesObj, Object alphaObj, Object gammaObj, Object epsilonObj, Object rewardObj) {
    
    // ensure that the right datatypes are used
    Integer episodes = Integer.valueOf(episodesObj.toString());
    Double alpha = Double.valueOf(alphaObj.toString());
    Double gamma = Double.valueOf(gammaObj.toString());
    Double epsilon = Double.valueOf(epsilonObj.toString());
    Integer reward = Integer.valueOf(rewardObj.toString());

    System.out.println("reward: " + reward);

    // List<Object> goalState = new ArrayList<>(Arrays.asList(goalDescription));
    // System.out.println("goalDescription[0]: " + goalDescription[0]);

    // List<Object> goalState = new ArrayList<>();

    // System.out.println("goalDescription[0].getClass().getName()" + goalDescription[0].getClass().getName());
    // goalState.add(2);
    // goalState.add(3);

    List<Object> goalState = Arrays.asList(((Byte)goalDescription[0]).intValue(), ((Byte)goalDescription[1]).intValue());


    // initialize Q(s,a) arbitrarily
    double[][] qTable = this.initializeQTable();
    
    // loop for each episode
    for (int i = 0; i < episodes; i++) {

      // // initialize s_t
      // this.lab.readCurrentState();
      // // random actions
      // Random rand = new Random();
      // int randomAction = rand.nextInt(8);
      // this.lab.performAction(randomAction);

      // initialize s_t again
      int currentState = this.lab.readCurrentState();

      // loop for each step of episode
      while(!this.lab.getCompatibleStates(goalState).contains(currentState)) {
        
        System.out.println("goalState"+ goalState);

        System.out.println("this.lab.getCompatibleStates(goalState): " + this.lab.getCompatibleStates(goalState));

        //  choose a from s using policy derived from Q (e.g. epsilon-greedy)
        List<Integer> actions = this.lab.getApplicableActions(currentState);
        System.out.println("actions: " + actions);
        double[] actionList = new double[actions.size()];
        for (int j = 0; j < actions.size(); j++) {
          actionList[j] = qTable[currentState][actions.get(j)];
        }
        
        int maxAction = (int) Arrays.stream(actionList).max().getAsDouble();
        
        // take action a, observe r, s'
        this.lab.performAction(maxAction);

        // s_t+1
        int newState = this.lab.readCurrentState();

        // Q(s,a) <- Q(s,a) + alpha[r + gamma * max Q(s',a') - Q(s,a)]
        qTable[currentState][maxAction] = qTable[currentState][maxAction] + alpha * (reward + gamma * qTable[newState][maxAction] - qTable[currentState][maxAction]);
        
        // s <- s'
        currentState = newState;

        System.out.println("Episode: " + i + " State: " + currentState + " Action: " + maxAction);

        // printQTable(qTable);


      }
    }
    this.qTables.put(goalState.hashCode(), qTable);
  }

/**
* Returns information about the next best action based on a provided state and the QTable for
* a goal description. The returned information can be used by agents to invoke an action 
* using a ThingArtifact.
*
* @param  goalDescription  the desired goal against the which the Q matrix is calculated (e.g., [2,3])
* @param  currentStateDescription the current state e.g. [2,2,true,false,true,true,2]
* @param  nextBestActionTag the (returned) semantic annotation of the next best action, e.g. "http://example.org/was#SetZ1Light"
* @param  nextBestActionPayloadTags the (returned) semantic annotations of the payload of the next best action, e.g. [Z1Light]
* @param nextBestActionPayload the (returned) payload of the next best action, e.g. true
**/
  @OPERATION
  public void getActionFromState(Object[] goalDescription, Object[] currentStateDescription,
      OpFeedbackParam<String> nextBestActionTag, OpFeedbackParam<Object[]> nextBestActionPayloadTags,
      OpFeedbackParam<Object[]> nextBestActionPayload) {

      double[][] qTable = this.qTables.get(Arrays.asList(goalDescription).hashCode());
      List<Integer> currentStates = this.lab.getCompatibleStates(Arrays.asList(currentStateDescription));
      List<List<Integer>> stateList = new ArrayList<>(this.lab.stateSpace);
      int currentState = stateList.indexOf(currentStates);

      double[] actions = qTable[currentState];
      int maxAction = (int) Arrays.stream(actions).max().getAsDouble();

      Action a = this.lab.actionSpace.get(maxAction);

      nextBestActionTag.set(a.getActionTag()); 
      Object payloadTags[] = a.getPayloadTags();
      nextBestActionPayloadTags.set(payloadTags);
      Object payload[] = a.getPayload();
      nextBestActionPayload.set(payload);
         
      //   // remove the following upon implementing Task 2.3!

      //   // sets the semantic annotation of the next best action to be returned 
      //   nextBestActionTag.set("http://example.org/was#SetZ1Light");

      //   // sets the semantic annotation of the payload of the next best action to be returned 
      //   Object payloadTags[] = { "Z1Light" };
      //   nextBestActionPayloadTags.set(payloadTags);

      //   // sets the payload of the next best action to be returned 
      //   Object payload[] = { true };
      //   nextBestActionPayload.set(payload);

      }

    /**
    * Print the Q matrix
    *
    * @param qTable the Q matrix
    */
  void printQTable(double[][] qTable) {
    System.out.println("Q matrix");
    for (int i = 0; i < qTable.length; i++) {
      System.out.print("From state " + i + ":  ");
     for (int j = 0; j < qTable[i].length; j++) {
      System.out.printf("%6.2f ", (qTable[i][j]));
      }
      System.out.println();
    }
  }

  /**
  * Initialize a Q matrix
  *
  * @return the Q matrix
  */
 private double[][] initializeQTable() {
    double[][] qTable = new double[this.stateCount][this.actionCount];
    for (int i = 0; i < stateCount; i++){
      for(int j = 0; j < actionCount; j++){
        qTable[i][j] = 0.0;
      }
    }
    return qTable;
  }
}
