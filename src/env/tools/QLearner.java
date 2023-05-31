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

    List<List<Integer>> stateList = new ArrayList<>(this.lab.stateSpace);

    // convert goal description to a list
    List<Object> goalState = Arrays.asList(((Byte)goalDescription[0]).intValue(), ((Byte)goalDescription[1]).intValue());

    // initialize Q(s,a) arbitrarily
    double[][] qTable = this.initializeQTable();
    
    // loop for each episode
    for (int i = 0; i < episodes; i++) {

      // initialize s_t
      this.lab.readCurrentState();
      // random actions
      Random rand = new Random();
      int randomAction = rand.nextInt(8);
      this.lab.performAction(randomAction);

      // initialize s_t again
      int currentState = this.lab.readCurrentState();

      // loop for each step of episode
      while(!this.lab.getCompatibleStates(goalState).contains(currentState)) {

        // choose a from s using policy derived from Q (e.g. epsilon-greedy)
        List<Integer> actions = this.lab.getApplicableActions(currentState);
        double[] actionsValue = new double[actions.size()];
        double maxValue = 0.0;
        int maxIndex = 0;
        for (int j = 0; j < actions.size(); j++) {
          actionsValue[j] = qTable[currentState][actions.get(j)];
          if (actionsValue[j] >= maxValue) {
            maxIndex = j;
          }
        }
        int maxAction = actions.get(maxIndex);

        // take action a, observe r, s'
        this.lab.performAction(maxAction);
         
        // s_t+1
        int newState = this.lab.readCurrentState();

        // get reward from s_t+1
        List<Integer> newStateList = stateList.get(newState);
        int newZ1Level = newStateList.get(0);
        int newZ2Level = newStateList.get(1);
        reward = newZ1Level + newZ2Level - (Integer) goalState.get(0) - (Integer) goalState.get(1);
        reward = 1/ (reward * -1)+1;

        // compute max Q(s',a')
        List<Integer> actionsNew = this.lab.getApplicableActions(newState);
        double[] actionsValueNew = new double[actionsNew.size()];
        double maxValueNew = 0.0;
        int maxIndexNew = 0;
        for (int j = 0; j < actions.size(); j++) {
          actionsValueNew[j] = qTable[newState][actions.get(j)];
          if (actionsValueNew[j] >= maxValueNew) {
            maxIndexNew = j;
          }
        }
        int maxActionNew = actions.get(maxIndexNew);

        // Q(s,a) <- Q(s,a) + alpha[r + gamma * max Q(s',a') - Q(s,a)]
        qTable[currentState][maxAction] = qTable[currentState][maxAction] + alpha * (reward + gamma * qTable[newState][maxActionNew] - qTable[currentState][maxAction]);
        
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

      // convert goal description to a list
      List<Object> goalState = Arrays.asList(((Byte)goalDescription[0]).intValue(), ((Byte)goalDescription[1]).intValue());
      double[][] qTable = this.qTables.get(goalState.hashCode());

      // convert current state description to a list
      List<Integer> currentStateList = new ArrayList<>();
      currentStateList.add(((Byte)currentStateDescription[0]).intValue());
      currentStateList.add(((Byte)currentStateDescription[1]).intValue());
      currentStateList.add(((Boolean)currentStateDescription[2]).booleanValue() ? 1 : 0);
      currentStateList.add(((Boolean)currentStateDescription[3]).booleanValue() ? 1 : 0);
      currentStateList.add(((Boolean)currentStateDescription[4]).booleanValue() ? 1 : 0);
      currentStateList.add(((Boolean)currentStateDescription[5]).booleanValue() ? 1 : 0);
      currentStateList.add(((Byte)currentStateDescription[6]).intValue());

      // get the index of the current state from the state space
      List<List<Integer>> stateList = new ArrayList<>(this.lab.stateSpace);
      int currentState = stateList.indexOf(currentStateList);

      // get the action with the highest Q value
      double[] actions = qTable[currentState];
      double maxValue = Arrays.stream(actions).max().getAsDouble();
      int actionIndex = -1;
      
      // Find the index of the maximum value
      for (int i = 0; i < actions.length; i++) {
          if (actions[i] == maxValue) {
              actionIndex = i;
              break;
          } else {
            System.err.println("Error: maxValue " + maxValue + " not found in actions array.");
          }
      } 

      Action a = this.lab.actionSpace.get(actionIndex);

      nextBestActionTag.set(a.getActionTag()); 
      Object payloadTags[] = a.getPayloadTags();
      nextBestActionPayloadTags.set(payloadTags);
      Object payload[] = a.getPayload();
      nextBestActionPayload.set(payload);

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
