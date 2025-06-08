# main.py
from collections import Counter
import random
import json
import re
import time
from typing import List
import openai
from Enums import Role, GameState
from DisplayAdapter import DisplayAdapter
import json
import os
from datetime import datetime
from experiencepool import ExperiencePool


# LLMPlayerBuilder 用于根据配置文件创建 LLMPlayer 实例
class LLMPlayerBuilder:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.api_configs = self.config['api_configs']
    def build_all(self, role: Role):
        players = []
        for config in self.api_configs:
            player = LLMPlayer(
                role=role,
                api_base=config['api_base'],
                model_name=config['model_name'],
                api_key=config['api_key']
            )
            players.append(player)
        return players

class Player:  # 玩家基类
    def __init__(self, role: Role):
        # 将所有属性都作为实例变量初始化，避免多个玩家共享同一状态
        self.role = role
        self.number = 0
        self.alive = True
        self.protected = False # 是否被守卫保护
        self.avatar = "./assets/default.png"
        self.display = None
        self.chatLog = []  # 每个玩家独有的聊天记录
        self.SavePotion = 1 # 女巫是否有解药
        self.KillPotion = 1 # 女巫是否有毒药
        self.dataCache = {}

    def requestSpeech(self, prompt) -> str:
        return self.display.input(prompt)

    def updateDisplay(self, data: dict):
        self.display.update(data)
        self.dataCache = data

    def requestVote(self, prompt: str) -> int:
        return int(self.display.input(prompt))

    def updateChat(self, sender: str, message: str):
        self.chatLog.append(f"{sender}: {message}")
        self.updateDisplay(self.dataCache)

    def updateSystem(self, message: str):
        self.updateChat("System", message)
        self.updateDisplay(self.dataCache)

class LLMPlayer(Player):
    def __init__(self, 
                 role: Role,
                 api_base: str,
                 model_name: str,
                 api_key: str,
                 temperature: float = 0.7,
                 max_retries: int = 3):
        super().__init__(role)
        self.api_base = api_base
        self.api_key = api_key
        # 按照你提供的方式初始化OpenAI客户端
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.memory = []  # 对话记忆
        self.questions = self._load_questions()# 加载问题库
        self.important_events = []  # 重要事件记录
        self.player_analysis = {}   # 玩家分析记录
        self.max_context_length = 2000  # 最大上下文长度
        self.experience_pool = ExperiencePool()

    def _load_questions(self):
        """加载问题库"""
        try:
            with open('question.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("警告：questions.json文件未找到，使用默认问题")
            return {
                "all": ["根据当前局势，你应该如何行动？"],
                "WEREWOLF": ["作为狼人，你的策略是什么？"],
                "VILLAGER": ["你怀疑谁是狼人？"],
                "SEER": ["你要查验谁？"],
                "WITCH": ["你要使用药水吗？"]
            }

    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        alive_players = [str(p.number) for p in self.game.getAlivePlayers()]
        role_desc = {
            Role.WEREWOLF: "你是狼人，夜晚与你的同伴讨论袭击目标，白天伪装成村民",
            Role.VILLAGER: "你是普通村民，通过观察找出狼人",
            Role.SEER: "你是预言家，每晚可以查验一名玩家的身份",
            Role.WITCH: "你是女巫，拥有解药和毒药各一次",
            Role.GUARD: "你是守卫，每晚可以守护一名玩家",
            Role.HUNTER: "你是猎人，若被放逐或夜间死亡，可以带走一名玩家",
        }
        base_prompt = f"""## 游戏规则（新的一局）
        你正在参与一场全新的狼人杀游戏，每局游戏互相独立，上一局的信息不再适用。
- 当前存活玩家：{', '.join(alive_players)}
- 你的角色：{role_desc[self.role]}
- 你的编号是{self.number}
- 游戏阶段：{self.game.state}
- 禁止暴露角色身份，除非你已经公开身份
- 你收到的内容就是从最近一次角色分配完成开始完整的游戏上下文流程和所有玩家发言记录 游戏刚开始时上下文为空时正常的
- 不要胡乱编造
- 用中文回复，保持自然对话语气
- 如果你是狼人，记住以下规则：1. 如果有人投票给你，你要记住他们；2. 预言家不能太早暴露；3. 如果你是狼人，要尽量误导村民...
- 如果你是狼人，请严格从 candidates 中选择一名玩家作为目标，禁止弃票（不能投 -1）
- 如果你是预言家，增加查验目标的随机性
- 好人阵营在白天投票时要更加激进，因为弃票会增加狼人获胜的概率
"""
        # 女巫专属提示
        if self.role == Role.WITCH:
            potion_status = []
            if self.SavePotion == 1:
                potion_status.append("解药可用")
            if self.KillPotion == 1:
                potion_status.append("毒药可用")
            potion_text = " | ".join(potion_status) if potion_status else "无药可用"
            return f"{base_prompt}\n- 当前药水状态：{potion_text}"
        return base_prompt
    
    def _extract_important_events(self):
        """提取重要事件"""
        important_events = []
        for log in self.chatLog:
            # 提取关键信息
            if any(keyword in log for keyword in [
                "死亡", "放逐", "遗言", "查验", "袭击", "毒杀", "解药", "守护","刀",
            ]):
                important_events.append(log)
            # 提取身份相关信息
            elif any(role in log for role in ["预言家", "女巫", "狼人", "猎人"," 神职", "村民"]):
                important_events.append(log)
        return important_events[-10:]  # 只保留最近10个重要事件
    
    def _summarize_player_behaviors(self):
        """总结玩家行为模式"""
        player_summaries = {}
        alive_players = [p.number for p in self.game.getAlivePlayers()]
        for player_num in alive_players:
            if player_num == self.number:
                continue 
            # 分析该玩家的发言和投票行为
            player_logs = [log for log in self.chatLog if f"玩家 {player_num}" in log]
            recent_logs = player_logs[-3:]  # 只看最近3次发言
            if recent_logs:
                summary = f"玩家{player_num}最近态度：{self._analyze_player_attitude(recent_logs)}"
                player_summaries[player_num] = summary
        
        return player_summaries
    
    def _analyze_player_attitude(self, logs):
        """分析玩家态度（简化版）"""
        aggressive_words = ["肯定是", "一定是", "必须投", "绝对","我注意到","可疑","怀疑"]
        defensive_words = ["不是我", "我觉得", "可能", "也许"]
        aggressive_count = sum(1 for log in logs for word in aggressive_words if word in log)
        defensive_count = sum(1 for log in logs for word in defensive_words if word in log)
        if aggressive_count > defensive_count:
            return "激进"
        elif defensive_count > aggressive_count:
            return "保守"
        else:
            return "中性"
    
    def _get_condensed_context(self) -> str:
        """获取压缩后的游戏上下文"""
        # 1. 重要事件
        important_events = self._extract_important_events()
        # 2. 玩家行为摘要
        player_summaries = self._summarize_player_behaviors()
        # # 3. 最近对话（只保留最近5轮）
        # recent_chat = self.chatLog[-5:] if len(self.chatLog) > 5 else self.chatLog
        # 4. 当前游戏状态
        game_status = f"""
## 当前状态
- 第{self.game.day}天，{self.game.state}
- 存活玩家：{[p.number for p in self.game.getAlivePlayers()]}
"""
        # 5. 组合上下文
        context = f"""{game_status}
## 重要事件回顾
{chr(10).join(important_events[-5:])}
## 玩家态度分析
{chr(10).join([f"- {summary}" for summary in player_summaries.values()])}
"""
        # 确保不超过最大长度
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "...[内容过长，已截断]"
        return context
    
    def _generate_dynamic_questions(self):
        """根据游戏状态动态生成问题"""
        dynamic_questions = []
        # 根据游戏天数生成问题
        if self.game.day == 1:
            dynamic_questions.append("第一天你需要特别注意什么？")
        elif self.game.day >= 3:
            dynamic_questions.append("游戏已经进行了几天，你从之前的发言中发现了什么规律？")
        # 根据存活人数生成问题
        alive_count = len(self.game.getAlivePlayers())
        if alive_count <= 5:
            dynamic_questions.append("现在人数较少，你的策略需要如何调整？")
        # 根据角色状态生成问题
        if self.role == Role.WITCH:
            if self.SavePotion == 0 and self.KillPotion == 0:
                dynamic_questions.append("你的药水都用完了，现在如何发挥作用？")
        return dynamic_questions
    
    def _get_random_questions(self, num_questions=2):
        """随机选择问题进行思考"""
        all_questions = self.questions.get("all", []) # 获取通用问题
        role_questions = self.questions.get(str(self.role), []) # 获取角色专属问题
        dynamic_questions = self._generate_dynamic_questions()
        # 合并问题池
        available_questions = all_questions + role_questions + dynamic_questions
        # 随机选择问题
        if len(available_questions) <= num_questions:
            return available_questions
        else:
            return random.sample(available_questions, num_questions)

    def _think_before_action(self):
        """在行动前进行思考（增强版，包含经验检索）"""
        # 获取当前上下文用于经验检索
        current_context = self._get_condensed_context()
        # 从经验池获取建议
        experience_advice = self.experience_pool.get_advice(
            current_context, str(self.role), "decision"
        )
        questions = self._get_random_questions(2)
        thinking_prompt = f"""## 行动前思考
在进行投票或发言之前，请先思考以下问题："""
        for i, question in enumerate(questions, 1):
            thinking_prompt += f"\n{i}. {question}"
        # 添加经验建议
        if experience_advice != "暂无相关经验可参考":
            thinking_prompt += f"\n\n## 历史经验参考\n{experience_advice}"
        thinking_prompt += "\n\n请综合考虑上述问题和历史经验，简要回答并说明你的行动计划："
        thinking_response = self._call_llm(thinking_prompt, is_print=True)
        think = thinking_prompt + thinking_response
        self.chatLog.append(f"[提问与思考] {think}")
        return think


    def _get_game_context(self) -> str:
        """获取聊天记录作为对话上下文"""
        context = self.chatLog
        return "\n".join(context)

    def _call_llm(self, prompt: str, is_print: bool) -> str:
        """调用LLM接口（新增流式处理但保持兼容性）"""
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        for _ in range(self.max_retries):
            try:
                # 初始化内容容器
                full_content = ""
                full_reasoning = ""
                # 创建流式请求
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=4098,  # 限制最大输出长度
                    stream=True  # 启用流式输出
                )
                # 实时处理流式响应
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    # 处理思维链内容
                    if getattr(delta, 'reasoning_content', None):
                        full_reasoning += delta.reasoning_content
                        # if is_print:
                        #     print(f"\033[90m{delta.reasoning_content}\033[0m", end="", flush=True)  # 灰色显示思维链
                    # 处理最终回答内容
                    if getattr(delta, 'content', None):
                        full_content += delta.content
                        if is_print:
                            print(delta.content, end="", flush=True)  # 正常显示回答内容
                print()  # 输出换行
                time.sleep(1)
                return full_content.strip()
            except Exception as e:
                print(f"\nAPI Error: {str(e)}")
                print(self.model_name, self.api_base, self.api_key)
                # 重试前清空已收集内容
                full_content = ""
                full_reasoning = ""
        return ""  # 维持失败返回空字符串

#     def requestSpeech(self, prompt: str) -> str:
#         """生成智能发言"""
#         thinking = self._think_before_action()
#         print(self._get_condensed_context())  # 调试输出
#         # 基于思考结果生成发言
#         full_prompt = f"""## 反思{thinking}
#                           ## 历史对话{self._get_game_context()} 
#                           ## 你的任务{prompt}
# 基于你刚才的思考，请用1-2句话进行发言，保持自然口语化，不要使用特殊符号。
# 注意：不要暴露你的思考过程，只说出你想让其他玩家听到的话。"""
#         response = self._call_llm(full_prompt, is_print=False)
#         clean_response = re.sub(r"【.*?】", "", response)
#         return clean_response[:100]
    def requestSpeech(self, prompt: str) -> str:
        """生成智能发言（增强版，包含经验指导）"""
        thinking = self._think_before_action()
        current_context = self._get_condensed_context()
        speech_advice = self.experience_pool.get_advice(
            current_context, str(self.role), "speech")
        full_prompt = f"""## 反思{thinking}## 历史对话{self._get_game_context()}## 你的任务{prompt}"""
        # 添加发言经验指导
        if speech_advice != "暂无相关经验可参考":
            full_prompt += f"\n\n## 发言经验参考\n{speech_advice}"
        full_prompt += "\n\n基于你的思考和经验参考，请用1-2句话进行发言，保持自然口语化，不要使用特殊符号。注意：不要暴露你的思考过程，只说出你想让其他玩家听到的话。"
        response = self._call_llm(full_prompt, is_print=False)
        clean_response = re.sub(r"【.*?】", "", response)
        return clean_response[:100]
        
    # def requestVote(self, prompt: str) -> int:
    #     """智能投票决策"""
    #     thinking = self._think_before_action()
    #     full_prompt = f"""## 反思{thinking}
    #                       ## 历史对话{self._get_game_context()}
    #                       ## 投票规则 {prompt}
    #     请严格按以下格式回复：{{"reason": "分析原因", "vote": 玩家编号或-1}}"""
    #     print(f"玩家{self.number}({self.role}):")  # 调试输出
    #     response = self._call_llm(full_prompt, is_print=True)
    #     try:
    #         if "{" in response:
    #             json_part = response[response.find("{"):response.find("}")+1]
    #             data = json.loads(json_part)
    #             return int(data["vote"])
    #         else:
    #             numbers = re.findall(r'\d+', response)
    #             return int(numbers[-1]) if numbers else -1
    #     except Exception:
    #         return -1
    def requestVote(self, prompt: str) -> int:
        """智能投票决策（增强版，包含经验指导）"""
        thinking = self._think_before_action()
        current_context = self._get_condensed_context()
        vote_advice = self.experience_pool.get_advice(
            current_context, str(self.role), "vote")
        full_prompt = f"""## 反思{thinking}## 历史对话{self._get_game_context()}## 投票规则{prompt}"""
        if vote_advice != "暂无相关经验可参考":
            full_prompt += f"\n\n## 投票经验参考\n{vote_advice}"
        full_prompt += '\n\n请综合考虑所有信息，严格按以下格式回复：{"reason": "分析原因", "vote": 玩家编号或-1}'
        print(f"玩家{self.number}({self.role}):")
        response = self._call_llm(full_prompt, is_print=True)
        try:
            if "{" in response:
                json_part = response[response.find("{"):response.find("}")+1]
                data = json.loads(json_part)
                return int(data["vote"])
            else:
                numbers = re.findall(r'\d+', response)
                return int(numbers[-1]) if numbers else -1
        except Exception:
            return -1

    def updateDisplay(self, data: dict):
        """同步游戏状态，不覆盖game对象引用"""
        # if self.role == Role.WITCH:
        #     self.hasSavePotion = data.get("hasSave", True)
        #     self.hasKillPotion = data.get("hasKill", True)
        # super().updateDisplay(data)

class Game:
    def __init__(self, players: List[Player]):
        self.day = 0
        self.dayLog = []
        self.players = players
        self.state = GameState.NIGHT
        self.night_deaths = []  # 用于保存夜间死亡玩家的编号
        total_players = len(players)
        if total_players < 5:
            raise ValueError("游戏需要至少5名玩家")
        # 确定特殊角色配置
        special_roles = [Role.SEER, Role.HUNTER]
        if total_players >= 8:
            special_roles.append(Role.WITCH)
        # 计算狼人数量
        max_werewolves = total_players - len(special_roles) - 1  # 至少保留1个村民
        werewolf_count = max(1, min(total_players // 3, max_werewolves))
        # 计算村民数量
        villager_count = total_players - werewolf_count - len(special_roles)
        if villager_count < 1:
            raise ValueError("角色分配失败：村民数量不足")
        # 构建角色列表
        roles = (
            [Role.WEREWOLF] * werewolf_count +
            special_roles +
            [Role.VILLAGER] * villager_count
        )
        random.shuffle(roles)
        # 分配角色和编号
        for i, player in enumerate(players):
            player.number = i + 1
            player.role = roles[i]
            player.game = self  # 绑定游戏实例
            player.alive = True  # 重置存活状态
            player.protected = False
            if hasattr(player, 'last_guarded'):
                player.last_guarded = None  # 重置守卫记忆
        print("角色分配完成：")
        for p in players:
            print(f"玩家 {p.number} 号：{p.role}")

    def save_chat_logs(self):
        """游戏结束后保存每个玩家的聊天记录到txt文件"""
        # 创建logs目录
        logs_dir = "chat_logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        # 生成时间戳作为文件夹名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_dir = os.path.join(logs_dir, f"game_{timestamp}")
        os.makedirs(game_dir)
        # 为每个玩家保存聊天记录
        for player in self.players:
            filename = f"player_{player.number}_{player.role}.txt"
            filepath = os.path.join(game_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"玩家 {player.number} 号聊天记录\n")
                f.write(f"角色: {player.role}\n")
                f.write(f"最终状态: {'存活' if player.alive else '死亡'}\n")
                f.write("=" * 50 + "\n\n")
                # 写入聊天记录
                for i, message in enumerate(player.chatLog, 1):
                    f.write(f"{i:3d}. {message}\n")
                # 添加统计信息
                f.write("\n" + "=" * 50 + "\n")
                f.write(f"总消息数: {len(player.chatLog)}\n")
        # 生成游戏总结文件
        summary_file = os.path.join(game_dir, "game_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"狼人杀游戏总结 - {timestamp}\n")
            f.write("=" * 50 + "\n")
            f.write(f"游戏天数: {self.day}\n")
            f.write(f"最终状态: {self.state}\n\n")
            # 角色分配
            f.write("角色分配:\n")
            for player in self.players:
                status = "存活" if player.alive else "死亡"
                f.write(f"  玩家 {player.number}: {player.role} ({status})\n")
            # 胜负统计
            alive_werewolves = [p for p in self.players if p.role == Role.WEREWOLF and p.alive]
            f.write(f"\n存活狼人数: {len(alive_werewolves)}\n")
            if len(alive_werewolves) == 0:
                f.write("游戏结果: 村民阵营胜利\n")
            else:
                f.write("游戏结果: 狼人阵营胜利\n")
        print(f"\n聊天记录已保存到: {game_dir}")
        return game_dir


    def checkWin(self) -> bool:
        """检查游戏是否结束"""
        alive_players = self.getAlivePlayers()
        werewolves = [p for p in alive_players if p.role == Role.WEREWOLF]
        villagers = [p for p in alive_players if p.role not in (Role.WEREWOLF, Role.SEER, Role.WITCH, Role.GUARD, Role.HUNTER)]
        special_roles = [p for p in alive_players if p.role in (Role.SEER, Role.WITCH, Role.GUARD, Role.HUNTER)]
        if not werewolves:
            self._broadcast("村民阵营胜利！")
            # self.updateDisplay()
            # 游戏结束时保存聊天记录
            self.save_chat_logs()
            return True
        if not villagers or not special_roles:
            self._broadcast("狼人阵营胜利！")
            # self.updateDisplay()
            # 游戏结束时保存聊天记录
            self.save_chat_logs()
            return True
        return False

    def getPlayer(self, number: int) -> Player:
        for player in self.players:
            if player.number == number:
                return player
        return None

    def getAlivePlayers(self) -> list:
        return [p for p in self.players if p.alive]

    def getAliveWerewolves(self) -> list:
        return [p for p in self.players if p.role == Role.WEREWOLF and p.alive]

    def getAliveWitch(self) -> Player:
        for p in self.getAlivePlayers():
            # print(f"Checking player {p.number} with role {p.role}")
            if p.role == Role.WITCH:
                return p
        return None
    
    def getAliveSeer(self) -> Player:
        for p in self.getAlivePlayers():
            # print(f"Checking player {p.number} with role {p.role}")
            if p.role == Role.SEER:
                return p
        return None

    def getAliveGuards(self) -> list:
        return [p for p in self.players if p.role == Role.GUARD and p.alive]

    def getHunters(self) -> list:
        return [p for p in self.players if p.role == Role.HUNTER]

    # def updateDisplay(self):
    #     for player in self.players:
    #         data = {
    #             "day": self.day,
    #             "state": self.state,
    #             "alivePlayers": [p.number for p in self.getAlivePlayers()],
    #             "deadPlayers": [p.number for p in self.players if not p.alive],
    #             "chatLog": player.chatLog,   # 每个玩家独有的聊天记录
    #             "role": player.role,
    #             "hasSave": player.hasSavePotion,
    #             "hasKill": player.hasKillPotion,
    #             "aiPlayers": {p.number: {"role": p.role, "model_name": p.model_name}
    #                       for p in self.players if isinstance(p, LLMPlayer)},
    #             "number": player.number
    #         }
    #         player.updateDisplay(data)

    def _broadcast(self, message: str, role_filter=None):
        """
        广播消息给所有玩家或指定角色：
          - 若role_filter不为空，则只向该角色发送消息；
          - 否则向所有玩家发送。
        """
        recipients = self.players if role_filter is None else [p for p in self.players if p.role == role_filter]
        print(message)  # 控制台输出
        for p in recipients:
            p.updateSystem(message)

    def _safe_vote(self, player, prompt, valid_targets, allow_abstain=True):
        """安全的投票请求，确保投票结果在允许范围内"""
        while True:
            try:
                # if player.role == Role.WITCH:
                #     print("witch request vote")
                #     vote = player.witch_requestVote(prompt)
                # else:
                #     vote = player.requestVote(prompt)
                vote = player.requestVote(prompt)
                if vote in valid_targets or (allow_abstain and vote == -1):
                    return vote
                player.updateSystem(f"无效目标，请选择：{valid_targets}")
            except ValueError:
                player.updateSystem("请输入有效数字")

    def _resolve_votes(self, votes, action_name, is_public=False):
        """通用投票决议逻辑"""
        valid_votes = [v for v in votes.values() if v != -1]
        if not valid_votes:
            self._broadcast(f"[系统消息]本次{action_name}未达成共识")
            return None
        counter = Counter(valid_votes)
        max_count = counter.most_common(1)[0][1]
        candidates = [num for num, cnt in counter.items() if cnt == max_count]
        if len(candidates) > 1:
            msg = f"[系统消息]平票！本次{action_name}无人出局"
            result = None
        else:
            result = candidates[0]
            msg = f"[系统消息]达成共识选择玩家 {result}"
        if is_public:
            self._broadcast(msg)
        else:
            self._broadcast(msg, role_filter=Role.WEREWOLF)
        return result

    def _hunter_action(self):
        # self.updateDisplay()
        hunters = self.getHunters()
        for hunter in hunters:
            self._broadcast(f"[系统消息]=== 猎人 {hunter.number} 请睁眼 ===", role_filter=Role.HUNTER)
            target = self._safe_vote(
                hunter,
                "你必须要选择带走一名玩家（输入玩家编号）",
                valid_targets=[p.number for p in self.getAlivePlayers()],
                allow_abstain=False
            )
            if target:
                self.getPlayer(target).alive = False
                hunter_msg = f"[系统消息]猎人 {hunter.number} 带走了玩家 {target}"
                self._broadcast(hunter_msg, role_filter=Role.HUNTER)

    def _werewolf_action(self):
        # self.updateDisplay()
        votes = {}
        candidates = [p.number for p in self.getAlivePlayers() if p.role != Role.WEREWOLF]
        
        # 狼人内部讨论（仅向狼人广播）
        self._broadcast("[系统消息]=== 狼人请睁眼，现在是夜间讨论时间 ===", role_filter=Role.WEREWOLF)
        for wolf in self.getAliveWerewolves():
            teammates = [str(p.number) for p in self.getAliveWerewolves() if p != wolf]
            wolf.updateSystem(f"[狼人队友信息] 你的队友是：{', '.join(teammates) if teammates else '只有你一人'}")
            speech = wolf.requestSpeech("狼人队伍讨论(仅队友可见) 请发言:")
            self._broadcast(f"🐺【狼人 {wolf.number}号】: {speech}", role_filter=Role.WEREWOLF)
        
        # 狼人投票
        for wolf in self.getAliveWerewolves():
            # self.updateDisplay()
            vote = self._safe_vote(
                wolf,
                f"请选择袭击目标（存活玩家：{candidates}）,注意：狼人不能弃票，不能平票",
                valid_targets=candidates,
                allow_abstain=False
            )
            votes[wolf.number] = vote
        return self._resolve_votes(votes, "袭击")
    
    def _execute_player(self, number):
        player = self.getPlayer(number)
        player.alive = False
        self._broadcast(f"[系统消息]=== 玩家 {number}号 被放逐 ===")
        last_words = player.requestSpeech("请发表遗言")
        role_reveal = f"({player.role})" if hasattr(player, 'role') else ""
        self._broadcast(f"[系统消息]【玩家 {number}号{role_reveal} 遗言】: {last_words}")

    def _witch_action(self, attack_target):
        # self.updateDisplay()
        witch = self.getAliveWitch()
        target_player = self.getPlayer(attack_target)
        if not witch:
            print("没有存活的女巫，跳过女巫行动")
            target_player.alive = False
            # 不在夜晚直接广播死亡信息，而是记录下来等待天亮时公布
            self.night_deaths.append(attack_target)
            return
        
        target_player = self.getPlayer(attack_target)
        self._broadcast("[系统消息]=== 女巫请睁眼 ===", role_filter=Role.WITCH)
        witch_msg = f"玩家 {attack_target}号 正在遭受袭击"
        self._broadcast(witch_msg, role_filter=Role.WITCH)
        attack_saved = False
        if witch.SavePotion == 1:
            if self._safe_vote(witch, "女巫是否使用解药？（1: 是，0: 否）", [0, 1]) == 1:
                target_player.alive = True
                witch.SavePotion = 0
                save_msg = f"[系统消息]女巫使用了解药拯救玩家 {attack_target}号"
                self._broadcast(save_msg, role_filter=Role.WITCH)
                attack_saved = True
        if not attack_saved:
            target_player.alive = False
            # 不在夜晚直接广播死亡信息，而是记录下来等待天亮时公布
            self.night_deaths.append(attack_target)
        else: 
            self._broadcast(f"[系统消息]女巫已无解药", role_filter=Role.WITCH)
        # 女巫毒药行动
        if witch.KillPotion == 1:
            valid_targets = [p.number for p in self.getAlivePlayers()] + [-1]
            target = self._safe_vote(
                witch,
                "请选择要毒杀的玩家（-1 表示不使用）",
                valid_targets=valid_targets,
                allow_abstain=True
            )
            if target != -1:
                self.getPlayer(target).alive = False
                witch.KillPotion = 0
                self.night_deaths.append(target)
                kill_msg = f"女巫毒杀了玩家 {target}"
                self._broadcast(kill_msg, role_filter=Role.WITCH)
        else:
            self._broadcast("[系统消息]女巫已无毒药", role_filter=Role.WITCH)

    def _seer_action(self):
        # self.updateDisplay()
        seer = self.getAliveSeer()
        if not seer:
            print("没有存活的预言家，跳过预言家行动")
            return
        self._broadcast("[系统消息]=== 预言家请睁眼 ===", role_filter=Role.SEER)
        target = self._safe_vote(
            seer,
            "请选择要查验的玩家",
            valid_targets=[p.number for p in self.getAlivePlayers()],
            allow_abstain=False
        )
        if target:
            role_info = self.getPlayer(target).role
            if role_info == Role.WEREWOLF:
                role_info = "狼人"
            else:
                role_info = "好人"
            see_msg = f"[系统消息]你查验了玩家 {target} 的身份是：{role_info}"
            self._broadcast(see_msg, role_filter=Role.SEER)

    def _daytime_discussion(self):
        # self.updateDisplay()
        self._broadcast(f"[系统消息]第 {self.day} 天开始，白天讨论时间")
        for player in self.getAlivePlayers():
            # self.updateDisplay()
            speech = player.requestSpeech("请发表你的看法")
            self._broadcast(f"玩家 {player.number} 说：{speech}")

    def _daytime_voting(self):
        votes = {}
        candidates = [p.number for p in self.getAlivePlayers()]
        # self.updateDisplay()
        for voter in self.getAlivePlayers():
            vote = self._safe_vote(
                voter,
                f"请选择要放逐的玩家（存活玩家：{candidates}）",
                valid_targets=candidates + [-1],
                allow_abstain=True
            )
            votes[voter.number] = vote
        return self._resolve_votes(votes, "放逐", is_public=True)

    def updateDay(self):
        self.day += 1
        # 夜晚阶段
        self.state = GameState.NIGHT
        # 重置夜间死亡记录
        self.night_deaths = []
        # self._guard_action()
        self._seer_action()
        attack_target = self._werewolf_action()
        if attack_target is None:
            print("没有狼人行动，跳过女巫行动")
        else: 
            self._witch_action(attack_target)
        # 白天阶段
        self.state = GameState.DAY
        # 天亮时公布夜间死亡信息
        if self.night_deaths:
            msg = "[系统消息]昨晚死亡玩家：" + ", ".join(str(n) for n in self.night_deaths)
            self._broadcast(msg)
            for dead_player_id in self.night_deaths:
                player = self.getPlayer(dead_player_id)
                if player.role == Role.HUNTER:
                    self._broadcast(f"[系统消息]猎人 {dead_player_id} 昨晚死亡，触发猎人技能")
                    self._hunter_action()
        else:
            self._broadcast("[系统消息]昨晚无人死亡")
        # 清空夜间记录，防止影响下一晚
        self.night_deaths = []
        self._daytime_discussion()
        eliminated = self._daytime_voting()
        if eliminated:
            self._execute_player(eliminated)
            if self.getPlayer(eliminated).role == Role.HUNTER:
                self._broadcast(f"[系统消息]猎人 {eliminated} 被放逐，进行猎人行动")
                # 猎人行动
                self._hunter_action()
        else:
            self._broadcast("[系统消息]今日无人被放逐")
        # self.updateDisplay()
        return self.checkWin()

    def main(self):
        for player in self.players:
            player.display = DisplayAdapter(player.number, len(self.players))
        try:
            while not self.checkWin():
                # self.updateDisplay()
                if self.updateDay():
                    break
        except KeyboardInterrupt:
            print("\n游戏被中断，正在保存聊天记录...")
            self.save_chat_logs()
        except Exception as e:
            print(f"\n游戏发生错误: {e}")
            print("正在保存聊天记录...")
            self.save_chat_logs()
            raise

# 示例用法：仅一个真人玩家，其余均为 AI 玩家
builder = LLMPlayerBuilder('config.json')  # 创建 LLMPlayerBuilder 实例
players = [
    *builder.build_all(None),  # 使用 builder 创建所有 AI 玩家
    # Player(None)  # 由真人控制的玩家
]

game = Game(players)
game.main()