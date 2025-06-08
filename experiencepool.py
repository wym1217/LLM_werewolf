import json
import os
import re
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Enums import Role, GameState

class ExperiencePool:
    def __init__(self, experience_dir: str = "./chat_logs"):
        self.experience_dir = experience_dir
        self.experiences = []  # 存储所有经验
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.experience_vectors = None
        self.load_experiences()
    
    def load_experiences(self):
        """从历史聊天记录中加载经验"""
        if not os.path.exists(self.experience_dir):
            print("经验池目录不存在，创建空经验池")
            return
        
        for game_folder in os.listdir(self.experience_dir):
            game_path = os.path.join(self.experience_dir, game_folder)
            if os.path.isdir(game_path):
                self._extract_game_experiences(game_path)
        
        if self.experiences:
            # 构建TF-IDF向量
            contexts = [exp["context"] for exp in self.experiences]
            self.experience_vectors = self.vectorizer.fit_transform(contexts)
            print(f"加载了 {len(self.experiences)} 条经验")
    
    def _extract_game_experiences(self, game_path: str):
        """从单局游戏中提取经验"""
        # 读取游戏总结
        summary_file = os.path.join(game_path, "game_summary.txt")
        if not os.path.exists(summary_file):
            return
        
        game_info = self._parse_game_summary(summary_file)
        
        # 读取每个玩家的聊天记录
        for filename in os.listdir(game_path):
            if filename.startswith("player_") and filename.endswith(".txt"):
                player_file = os.path.join(game_path, filename)
                self._extract_player_experiences(player_file, game_info)
    
    def _parse_game_summary(self, summary_file: str) -> Dict:
        """解析游戏总结文件"""
        game_info = {"winner": None, "total_days": 0, "player_roles": {}}
        
        with open(summary_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 提取胜者
            if "村民阵营胜利" in content:
                game_info["winner"] = "villager"
            elif "狼人阵营胜利" in content:
                game_info["winner"] = "werewolf"
            
            # 提取游戏天数
            day_match = re.search(r"游戏天数: (\d+)", content)
            if day_match:
                game_info["total_days"] = int(day_match.group(1))
            
            # 提取角色分配
            lines = content.split('\n')
            for line in lines:
                if "玩家" in line and ":" in line:
                    match = re.search(r"玩家 (\d+): (\w+)", line)
                    if match:
                        player_num = int(match.group(1))
                        role = match.group(2)
                        game_info["player_roles"][player_num] = role
        
        return game_info
    
    def _extract_player_experiences(self, player_file: str, game_info: Dict):
        """从玩家聊天记录中提取经验"""
        with open(player_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析玩家信息
        lines = content.split('\n')
        player_num = None
        player_role = None
        
        for line in lines[:5]:  # 前几行包含玩家信息
            if "玩家" in line and "号聊天记录" in line:
                match = re.search(r"玩家 (\d+) 号", line)
                if match:
                    player_num = int(match.group(1))
            elif "角色:" in line:
                player_role = line.split("角色:")[-1].strip()
        
        if not player_num or not player_role:
            return
        
        # 提取经验片段
        self._extract_decision_experiences(lines, player_num, player_role, game_info)
        self._extract_speech_experiences(lines, player_num, player_role, game_info)
        self._extract_voting_experiences(lines, player_num, player_role, game_info)
    
    def _extract_decision_experiences(self, lines: List[str], player_num: int, 
                                    player_role: str, game_info: Dict):
        """提取决策相关经验"""
        thinking_pattern = re.compile(r'\[提问与思考\].*?你的行动计划：(.*?)(?=\n|$)')
        
        for i, line in enumerate(lines):
            if "[提问与思考]" in line:
                # 提取思考内容
                thinking_match = thinking_pattern.search(line)
                if thinking_match:
                    action_plan = thinking_match.group(1).strip()
                    
                    # 构建上下文
                    context_lines = lines[max(0, i-5):i+3]  # 前后几行作为上下文
                    context = " ".join([l.strip() for l in context_lines if l.strip()])
                    
                    experience = {
                        "type": "decision",
                        "role": player_role,
                        "context": context,
                        "action": action_plan,
                        "outcome": "win" if game_info["winner"] else "unknown",
                        "day": self._extract_day_from_context(context),
                        "game_phase": self._extract_phase_from_context(context)
                    }
                    self.experiences.append(experience)
    
    def _extract_speech_experiences(self, lines: List[str], player_num: int,
                                   player_role: str, game_info: Dict):
        """提取发言相关经验"""
        for i, line in enumerate(lines):
            if f"玩家 {player_num} 说：" in line:
                speech = line.split("说：")[-1].strip()
                
                # 构建上下文（发言前的情况）
                context_lines = lines[max(0, i-5):i]
                context = " ".join([l.strip() for l in context_lines if l.strip()])
                
                experience = {
                    "type": "speech",
                    "role": player_role,
                    "context": context,
                    "speech": speech,
                    "outcome": game_info["winner"],
                    "day": self._extract_day_from_context(context)
                }
                self.experiences.append(experience)
    
    def _extract_voting_experiences(self, lines: List[str], player_num: int,
                                   player_role: str, game_info: Dict):
        """提取投票相关经验"""
        vote_pattern = re.compile(r'reason.*?vote.*?(\d+)', re.IGNORECASE)
        
        for i, line in enumerate(lines):
            vote_match = vote_pattern.search(line)
            if vote_match:
                voted_player = int(vote_match.group(1))
                
                # 构建投票前的上下文
                context_lines = lines[max(0, i-5):i]
                context = " ".join([l.strip() for l in context_lines if l.strip()])
                
                experience = {
                    "type": "vote",
                    "role": player_role,
                    "context": context,
                    "vote_target": voted_player,
                    "outcome": game_info["winner"],
                    "day": self._extract_day_from_context(context)
                }
                self.experiences.append(experience)
    
    def _extract_day_from_context(self, context: str) -> int:
        """从上下文中提取游戏天数"""
        day_match = re.search(r'第 (\d+) 天', context)
        return int(day_match.group(1)) if day_match else 1
    
    def _extract_phase_from_context(self, context: str) -> str:
        """从上下文中提取游戏阶段"""
        if "夜间" in context or "请睁眼" in context:
            return "night"
        elif "白天" in context or "讨论" in context:
            return "day"
        return "unknown"
    
    def retrieve_relevant_experiences(self, current_context: str, role: str, 
                                    experience_type: str = None, top_k: int = 3) -> List[Dict]:
        """检索相关经验"""
        if not self.experiences or self.experience_vectors is None:
            return []
        
        # 过滤同角色经验
        filtered_experiences = []
        for i, exp in enumerate(self.experiences):
            if exp["role"] == role:
                if experience_type is None or exp["type"] == experience_type:
                    filtered_experiences.append((i, exp))
        
        if not filtered_experiences:
            return []
        
        # 计算相似度
        current_vector = self.vectorizer.transform([current_context])
        indices = [i for i, _ in filtered_experiences]
        relevant_vectors = self.experience_vectors[indices]
        
        similarities = cosine_similarity(current_vector, relevant_vectors)[0]
        
        # 获取最相似的经验
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_experiences = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # 相似度阈值
                exp = filtered_experiences[idx][1]
                exp["similarity"] = similarities[idx]
                relevant_experiences.append(exp)
        
        return relevant_experiences
    
    def get_advice(self, current_context: str, role: str, action_type: str) -> str:
        """基于历史经验提供建议"""
        experiences = self.retrieve_relevant_experiences(
            current_context, role, action_type, top_k=5
        )
        
        if not experiences:
            return "暂无相关经验可参考"
        
        # 分析成功经验
        successful_experiences = [exp for exp in experiences if exp["outcome"] == "win"]
        
        advice_parts = []
        
        if successful_experiences:
            advice_parts.append("### 成功经验借鉴：")
            for exp in successful_experiences[:2]:  # 取前2个成功经验
                if action_type == "speech" and "speech" in exp:
                    advice_parts.append(f"- 相似情况下曾成功发言：'{exp['speech']}'")
                elif action_type == "decision" and "action" in exp:
                    advice_parts.append(f"- 相似情况下的成功策略：{exp['action']}")
                elif action_type == "vote" and "vote_target" in exp:
                    advice_parts.append(f"- 相似情况下投票给了玩家 {exp['vote_target']}")
        
        # 统计建议
        if action_type == "vote":
            vote_targets = [exp["vote_target"] for exp in experiences if "vote_target" in exp]
            if vote_targets:
                from collections import Counter
                common_targets = Counter(vote_targets).most_common(2)
                advice_parts.append(f"### 投票倾向分析：")
                for target, count in common_targets:
                    advice_parts.append(f"- 玩家 {target} 被投票 {count} 次")
        
        return "\n".join(advice_parts) if advice_parts else "经验数据不足，请谨慎决策"