"""
In-Context Learning with external policy

Current Flow:
1. Collect N demonstrations via teleoperation
2. Send demonstrations to LLM (OpenAI API) as context
3. LLM returns an adapted policy/trajectory
4. Execute the LLM-generated trajectory
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field

import numpy as np
from openai import OpenAI

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import make_default_processors
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class InContextConfig:
    """Configuration for simple context learning."""
    
    # Basic setup
    # Basic setup
    robot: RobotConfig = field(default_factory=lambda: so101_follower.SO101FollowerConfig(port="/dev/ttyACM1"))
    teleop: TeleoperatorConfig = field(default_factory=lambda: so101_leader.SO101LeaderConfig(port="/dev/ttyACM0"))
    control_robot: bool = True # Whether to send actions to the robot
    display_data: bool = False  # Whether to display data in Rerun
    
    # Parameters
    fps: int = 30
    num_demonstrations: int = 1  # N demonstrations to collect
    demo_duration_s: int = 10  # How long to demonstrate
    exec_duration_s: int = 10  # How long to execute (can match demo_duration_s)
    
    # LLM settings
    openai_api_key: str | None = None  # OpenAI API key (or set OPENAI_API_KEY env var)
    # assert openai_api_key, "OpenAI API key must be set!"
    llm_model: str = "gpt-4o-mini-2024-07-18"  # OpenAI model to use
    llm_prompt: str = "You are a robot control policy. Given demonstration trajectories, output an adapted trajectory."

class InContextLearning:
    """In-context learning: collect demos, send to LLM, execute LLM output."""

    def __init__(self, cfg: InContextConfig):
        """Initialize the session."""
        self.cfg = cfg
        
        # Initialize robot and teleoperator
        self.robot = make_robot_from_config(cfg.robot)
        self.teleop = make_teleoperator_from_config(cfg.teleop)
        
        # Initialize processors
        (
            self.teleop_action_processor,
            self.robot_action_processor,
            self.robot_observation_processor,
        ) = make_default_processors()
        
        # Initialize OpenAI client
        if cfg.openai_api_key:
            self.llm_client = OpenAI(api_key=cfg.openai_api_key)
        else:
            self.llm_client = None
        
        # Connect devices
        logging.info("Connecting robot and teleoperator...")
        self.robot.connect()
        self.teleop.connect()
        
        if not self.robot.is_connected or not self.teleop.is_connected:
            raise ValueError("Failed to connect robot or teleoperator!")
        
        # Keyboard listener
        self.listener, self.events = init_keyboard_listener()
        
        # Initialize visualization
        if cfg.display_data:
            init_rerun(session_name="simple_context_learning")
        
        logging.info("Simple context learning initialized successfully")
        # Initialize visualization
        if cfg.display_data:
            init_rerun(session_name="simple_context_learning")

    def run(self):
        """Main execution: collect demos → LLM processes → execute LLM output."""
        
        # Collect N demonstrations
        log_say(f"Collecting {self.cfg.num_demonstrations} demonstration(s)")
        demonstrations = self._collect_demonstrations()
        
        if not demonstrations:
            log_say("No demonstrations collected. Exiting.")
            return
        
        log_say(f"Collected {len(demonstrations)} demonstration(s)")

        # # Policy selects one demonstration randomly - debugging
        # selected_demo = random.choice(demonstrations)
        # demo_idx = demonstrations.index(selected_demo)
        # log_say(f"Policy randomly selected demonstration {demo_idx + 1}")

        # Send demonstrations to LLM and get adapted policy
        # log_say("Sending demonstrations to LLM for processing...")
        # adapted_trajectory = self._query_llm_for_policy(demonstrations)
        
        # if not adapted_trajectory:
        #     log_say("LLM did not return a valid trajectory. Falling back to random selection.")
        #     adapted_trajectory = random.choice(demonstrations)
        
        # # # Execute the selected demonstration - debugging
        # # log_say("Executing selected demonstration")
        # 
        # # Step 3: Execute the LLM-adapted trajectory
        # log_say("Executing LLM-adapted trajectory")
        # self._execute_demonstration(adapted_trajectory)
                
        log_say("Execution complete!")
    
    def _collect_demonstrations(self) -> list[list[dict]]:
        """
        Collect N demonstrations via teleoperation.
        
        Returns:
            List of demonstrations, where each demonstration is a list of
            (observation, action) pairs.
        """
        demonstrations = []
        
        for demo_idx in range(self.cfg.num_demonstrations):
            log_say(f"Recording demonstration {demo_idx + 1}/{self.cfg.num_demonstrations}")
            log_say("Use teleoperation to demonstrate the task")
            
            demo_trajectory = []
            t0 = time.perf_counter()
            
            # Keyframe tracking
            prev_robot_action = None
            g_keys, j_keys = None, None
            
            while time.perf_counter() - t0 < self.cfg.demo_duration_s:
                if self.events["exit_early"] or self.events["stop_recording"]:
                    log_say("Demonstration interrupted")
                    break
                
                loop_t0 = time.perf_counter()
                
                # Get observation
                obs = self.robot.get_observation()
                obs_processed = self.robot_observation_processor(obs)
                
                # Get teleop action
                action = self.teleop.get_action()
                action_processed = self.teleop_action_processor((action, obs))
                
                # Execute action
                robot_action = self.robot_action_processor((action_processed, obs_processed))
                if self.cfg.control_robot:
                    self.robot.send_action(robot_action)

                # Real-time keyframe detection
                is_keyframe = False
                if prev_robot_action is None:
                    is_keyframe = True # First frame
                    keys = list(robot_action.keys())
                    g_keys = [k for k in keys if "gripper" in k.lower()]
                    j_keys = [k for k in keys if k not in g_keys]
                else:
                    # Gripper change
                    if any(abs(robot_action[k] - prev_robot_action[k]) > 1e-3 for k in g_keys):
                        is_keyframe = True
                    # Low velocity (0.01 threshold)
                    # elif np.linalg.norm([robot_action[k] - prev_robot_action[k] for k in j_keys]) < 0.01:
                    #     is_keyframe = True
                
                if is_keyframe and prev_robot_action is not None:
                     print("Keyframe detected")

                # Store in trajectory
                demo_trajectory.append({
                    "observation": obs_processed,
                    "action": action_processed,
                    "robot_action": robot_action,
                    "keyframe": is_keyframe
                })
                
                prev_robot_action = robot_action
                
                # Visualize
                if self.cfg.display_data:
                    log_rerun_data(
                        observation=obs_processed,
                        action=action_processed,
                    )
                
                # Maintain FPS
                dt_s = time.perf_counter() - loop_t0
                precise_sleep(max(1.0 / self.cfg.fps - dt_s, 0.0))
            
            if demo_trajectory:
                # Mark last frame as keyframe
                demo_trajectory[-1]["keyframe"] = True
                demonstrations.append(demo_trajectory)
                logging.info(
                    f"Demonstration {demo_idx + 1} recorded: {len(demo_trajectory)} frames"
                )
            
            # Pause between demonstrations if collecting multiple
            if demo_idx < self.cfg.num_demonstrations - 1:
                log_say("Prepare for the next demonstration...")
                time.sleep(5.0)
                
        return demonstrations
    
    def _query_llm_for_policy(self, demonstrations: list[list[dict]]) -> list[dict]:
        """
        Send demonstrations to LLM and get an adapted trajectory.
        
        Args:
            demonstrations: List of demonstration trajectories
            
        Returns:
            Adapted trajectory (list of action dicts)
        """
        # Convert demonstrations to a format suitable for LLM
        demo_summary = self._format_demonstrations_for_llm(demonstrations)
        
        # Create the prompt
        system_prompt = self.cfg.llm_prompt
        user_prompt = f"""I have collected {len(demonstrations)} demonstration(s) of a robot task.

        Demonstration data:
        {demo_summary}

        Based on these demonstrations, please select the best demonstration to execute or describe how to adapt them. 
        For now, simply respond with the index of the demonstration you think is best (0-indexed), or -1 to average them.

        Respond in JSON format: {{"selected_demo_index": <int>, "reasoning": "<your reasoning>"}}"""
        try:
            # Query the LLM
            logging.info("Querying LLM...")
            response = self.llm_client.chat.completions.create(
                model=self.cfg.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
            )
            
            llm_output = response.choices[0].message.content
            logging.info(f"LLM response: {llm_output}")
            llm_output = llm_output.strip()
            # Parse the LLM's response
            result = json.loads(llm_output)
            selected_idx = result.get("selected_demo_index", 0)
            reasoning = result.get("reasoning", "No reasoning provided")
            
            log_say(f"LLM selected demonstration {selected_idx}: {reasoning}")
            
            # Return the selected demonstration or averaged version
            if selected_idx == -1:
                log_say("LLM requested averaging demonstrations")
                return self._average_demonstrations(demonstrations)
            elif 0 <= selected_idx < len(demonstrations):
                return demonstrations[selected_idx]
            else:
                log_say(f"Invalid index {selected_idx}, defaulting to first demonstration")
                return demonstrations[0]
                
        except Exception as e:
            logging.error(f"Error querying LLM: {e}")
            return None
    
    def extract_keyframes(self, trajectory: list[dict], vel_thresh: float = 0.01) -> list[int]:
        """Extract keyframes: gripper changes or low velocity."""
        if not trajectory: return []
        idxs = {0, len(trajectory) - 1}
        act_key = "robot_action"  
        keys = list(trajectory[0][act_key].keys())
        g_keys = [k for k in keys if "gripper" in k.lower()]
        j_keys = [k for k in keys if k not in g_keys]
        
        for i in range(1, len(trajectory) - 1):
            curr, prev = trajectory[i][act_key], trajectory[i-1][act_key]
            # Gripper change
            if any(abs(curr[k] - prev[k]) > 1e-3 for k in g_keys):
                idxs.add(i); continue
            # Low velocity
            vel = np.linalg.norm([curr[k] - prev[k] for k in j_keys])
            if vel < vel_thresh: idxs.add(i)
                
        return sorted(list(idxs))
    
    def _format_demonstrations_for_llm(self, demonstrations: list[list[dict]]) -> str:
        """
        Format demonstrations into a readable summary for the LLM.
        
        Args:
            demonstrations: List of demonstration trajectories
            
        Returns:
            Formatted string describing the demonstrations
        """
        summary_lines = []
        
        for demo_idx, demo in enumerate(demonstrations):
            # Get basic stats about the demonstration
            num_steps = len(demo)
            
            # Sample a few actions to show the LLM
            sample_indices = [0, num_steps // 2, num_steps - 1] if num_steps > 2 else range(num_steps)
            
            summary_lines.append(f"\nDemonstration {demo_idx}:")
            summary_lines.append(f"  - Total steps: {num_steps}")
            summary_lines.append(f"  - Sample actions:")
            
            for idx in sample_indices:
                if idx < num_steps:
                    action = demo[idx]["action"]
                    # Format action as a simple string
                    action_str = ", ".join([f"{k}: {v:.3f}" for k, v in list(action.items())[:5]])
                    summary_lines.append(f"    Step {idx}: {action_str}")
        
        return "\n".join(summary_lines)
    
    def _average_demonstrations(self, demonstrations: list[list[dict]]) -> list[dict]:
        """
        Average multiple demonstrations into a single trajectory.
        
        Args:
            demonstrations: List of demonstration trajectories
            
        Returns:
            Averaged trajectory
        """
        if len(demonstrations) == 1:
            return demonstrations[0]
        
        # Find the minimum length
        min_length = min(len(demo) for demo in demonstrations)
        
        averaged_trajectory = []
        
        for step_idx in range(min_length):
            # Collect all actions at this timestep
            actions_at_step = [demo[step_idx] for demo in demonstrations]
            
            # Average the robot actions (numerical values)
            avg_robot_action = {}
            first_robot_action = actions_at_step[0]["robot_action"]
            
            for key in first_robot_action.keys():
                values = [action["robot_action"][key] for action in actions_at_step]
                avg_robot_action[key] = np.mean(values)
            
            # Use the observation and processed action from the first demo as reference
            averaged_trajectory.append({
                "observation": actions_at_step[0]["observation"],
                "action": actions_at_step[0]["action"],
                "robot_action": avg_robot_action,
            })
        
        logging.info(f"Averaged {len(demonstrations)} demonstrations into {len(averaged_trajectory)} steps")
        return averaged_trajectory
    
    def _execute_demonstration(self, demonstration: list[dict]):
        """
        Execute a selected demonstration trajectory.
        
        Args:
            demonstration: List of (observation, action) pairs to replay
        """
        log_say(f"Executing trajectory with {len(demonstration)} steps")
        
        # Simple replay: just execute the recorded actions
        t0 = time.perf_counter()
        executed_steps = 0
        
        for step_idx, step in enumerate(demonstration):
            if self.events["exit_early"] or self.events["stop_recording"]:
                log_say("Execution interrupted")
                break
            
            # Check if we've exceeded execution duration
            if time.perf_counter() - t0 >= self.cfg.exec_duration_s:
                break
            
            loop_t0 = time.perf_counter()
            
            # Get current observation (for context, though not used in simple version)
            current_obs = self.robot.get_observation()
            current_obs_processed = self.robot_observation_processor(current_obs)
            
            # Use the stored action from the demonstration
            robot_action = step["robot_action"]
            
            # Execute the action
            self.robot.send_action(robot_action)
            executed_steps += 1
            
            # Visualize
            if self.cfg.display_data:
                log_rerun_data(
                    observation=current_obs_processed,
                    action=step["action"],
                )
            
            # Maintain FPS
            dt_s = time.perf_counter() - loop_t0
            precise_sleep(max(1.0 / self.cfg.fps - dt_s, 0.0))
        
        logging.info(
            f"Execution completed: {executed_steps}/{len(demonstration)} steps executed"
        )
    
    def cleanup(self):
        """Clean up resources."""
        logging.info("Cleaning up...")
        try:
            self.robot.disconnect()
        except Exception as e:
            logging.warning(f"Failed to disconnect robot cleanly: {e}")
        
        try:
            self.teleop.disconnect()
        except Exception as e:
            logging.warning(f"Failed to disconnect teleop cleanly: {e}")
            
        self.listener.stop()


@parser.wrap()
def main(cfg: InContextConfig):
    """Main entry point."""
    init_logging()
    
    logging.info(f"Configuration: N={cfg.num_demonstrations} demonstrations")
    
    # Initialize and run
    session = InContextLearning(cfg)
    
    try:
        session.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        session.cleanup()


if __name__ == "__main__":
    main()
