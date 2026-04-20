代码库导览

你希望我先做一个“代码库导览”。我会先快速读取核心文档和入口文件（README、CLI、编排核心），然后给你一份结构化的上手介绍：做什么、怎么跑、关键模块、典型工作流。

这是一个“自动科研流水线”项目：给定研究主题后，系统会按固定阶段完成文献调研、想法辩论、实验执行、写作整合、LaTeX 出稿和复盘演化。整体定位在 README.md。

**核心定位**
- 目标：从 topic 到论文的端到端自动化（含 GPU 实验与写作）。
- 计算后端：RunPod，生命周期管理在 runpod_backend.py。
- 运行方式：CLI + Claude 命令体系并存，入口在 cli.py。

**你先看这 5 个文件就够上手**
- 项目全景与命令： README.md
- CLI 入口与命令映射： cli.py
- 主编排器（API surface）： orchestrate.py
- 阶段流转规则（状态机）： state_machine.py
- 工作区文件协议（系统“共享内存”）： workspace.py

**系统怎么跑起来（执行链路）**
- 用户通过 CLI/命令触发后，先进入 orchestrator（orchestrate.py）。
- orchestrator 调用 lifecycle 生成当前阶段 Action（lifecycle.py）。
- 每个阶段完成后，state machine 根据结果和分数决定下一阶段（state_machine.py）。
- 阶段列表是固定 20 阶段，定义在 constants.py。

**数据与状态怎么组织**
- 所有 agent 通过 workspace 文件通信，不走进程内共享对象，核心在 workspace.py。
- 当前阶段、迭代次数、错误等写在 status.json 语义模型中（同文件里 WorkspaceStatus 定义）。
- 这种“文件即协议”的设计使得可审计、可恢复、易做断点续跑。

**你会最常用的能力**
- 状态查看、初始化、实验调度、演化、self-heal、latex 编译都在 cli.py。
- 实验执行与 pod 管理（创建、SSH、上传、远程执行、停止）在 runpod_backend.py。

**建议的阅读顺序**
1. README.md
2. cli.py
3. orchestrate.py
4. lifecycle.py
5. state_machine.py
6. workspace.py
7. runpod_backend.py

如果你愿意，我下一步可以给你一版“新人 30 分钟上手路径”，按你当前目标（改功能/修 bug/跑通全流程）定制到具体命令和文件。

---

可以。用“5岁小朋友版”来讲这个项目：

把它想成一个“会自己写论文的机器人实验室”。

1. 你给它一个题目  
2. 它先去看别人做过什么  
3. 它自己想点子、做小实验、大实验  
4. 它把结果写成论文  
5. 不满意就再来一轮，满意就结束

最核心的几位“角色”是：

1. 接待员（你和系统说话的入口）  
见 cli.py

2. 总指挥（决定下一步做什么）  
见 orchestrate.py

3. 流程红绿灯（什么时候前进、什么时候回头）  
见 state_machine.py

4. 公共白板（所有智能体都往这里写文件交流）  
见 workspace.py

5. GPU 司机（去 RunPod 开机器、传代码、跑实验、收结果）  
见 runpod_backend.py

6. 说明书（全局地图）  
见 README.md

你现在打开的 workspace.py 可以理解成“实验室秘书”：
- 记住当前做到哪一步（stage、iteration）
- 读写 JSON 和文本文件
- 新建下一轮迭代目录
- 清理临时产物但保留经验总结

给你的“30 分钟超快上手路线（ELI5）”：

1. 先看 8 分钟  
快速浏览 README.md 的 Architecture、Pipeline、Quick Start

2. 再看 7 分钟  
看 cli.py 里有哪些命令（status、init、experiment-run 等）

3. 再看 8 分钟  
看 orchestrate.py 和 lifecycle.py，理解“如何生成下一步 action”

4. 最后 7 分钟  
看 workspace.py 和 state_machine.py，理解“状态怎么存、怎么转”

如果你愿意，我下一条可以给你一份“看代码打卡清单”，直接按函数名带你逐个看（比如先看 Workspace.get_status，再看 Lifecycle.record_result），10 分钟内把主线彻底串起来。

---

太好了，你这个阶段最需要的是“抓主线”，不是学完 Python。

你可以把这个项目当成一个“自动科研公司”：

1. 你给任务（研究题目）
2. 总控安排流程（先调研，再实验，再写作）
3. 各小组通过文件协作
4. 最后产出论文和复盘

**先记住这 4 个角色（10 分钟入门）**
1. 前台命令入口：负责接收你输入的命令  
cli.py

2. 总控大脑：决定当前阶段该做什么  
orchestrate.py

3. 流程规则：决定下一步去哪一关（继续/回退/结束）  
state_machine.py

4. 公共文件板：所有 agent 都靠它读写状态和结果  
workspace.py

**你只要理解一个核心闭环**
1. 读状态：现在在第几阶段（stage）？
2. 生成动作：这阶段要执行什么 action？
3. 写结果：阶段完成后记录 result 和 score
4. 算下一步：状态机决定 next stage
5. 重复直到 done

这 5 步就是整个系统的“心跳”。

**你现在看的 workspace.py，到底在干嘛（小白版）**
- 它像“秘书”
- 记录项目当前状态（stage、iteration、errors）
- 提供读写文件能力（read_file、write_file、read_json、write_json）
- 开启新一轮迭代（new_iteration）
- 清理临时产物但保留经验总结（clear_iteration_artifacts）

换句话说：它不是“算法”，它是“项目管理和信息流基础设施”。

**如果你不熟 Python，只看这几个语法就够了**
1. class: 一组功能的打包（比如 Workspace）
2. def: 一个函数
3. self: 这个对象自己
4. Path: 文件路径工具
5. dict/list: JSON 对象/数组对应结构

你不需要先精通 Python，先看“函数名字 + 输入输出 + 何时调用”。

**20 分钟快速上手路线（适合 AI agent 新手）**
1. 5 分钟看全局说明  
README.md

2. 5 分钟看“入口到总控”  
cli.py  
orchestrate.py

3. 5 分钟看“状态怎么推进”  
lifecycle.py  
state_machine.py

4. 5 分钟看“文件怎么当协议”  
workspace.py

**你接下来最实用的学习方式**
1. 先别读全仓库，只沿着一条调用链追：cli -> orchestrate -> lifecycle -> state_machine -> workspace  
2. 每看到一个函数，只回答三件事：它读什么？写什么？下一步调谁？  
3. 把“阶段流转图”画在纸上，比啃代码快 3 倍

如果你愿意，我下一条可以给你一份“超短打卡版”：只看 12 个函数名，每个函数一句话解释，15 分钟就能把主干彻底搞懂。

---

太好了，给你一份“15 分钟打卡版”，专门给 Python/Agent 新手。

你只看这 12 个函数，就能抓住主线。

1. Workspace.get_status  
看项目当前在哪一关（stage）、第几轮（iteration）。  
文件: workspace.py

2. Workspace.save_status  
把状态写回 status.json，系统靠它“记住进度”。  
文件: workspace.py

3. Workspace.update_stage  
单纯切换到下一阶段。  
文件: workspace.py

4. Workspace.new_iteration  
开启新一轮研究循环（iteration +1），并准备目录。  
文件: workspace.py

5. Lifecycle.get_next_action  
根据当前 stage，决定“这一步该做什么动作”。  
文件: lifecycle.py

6. Lifecycle.record_result  
记录本阶段结果，然后触发“算下一步”。  
文件: lifecycle.py

7. StateMachine.natural_next_stage  
核心路由规则：继续、回退、pivot、done 都在这里判定。  
文件: state_machine.py

8. StateMachine.is_pipeline_done  
判断是否达到结束条件（质量阈值或最大迭代）。  
文件: state_machine.py

9. FarsOrchestrator.get_next_action  
把 lifecycle 产出的动作对外暴露给 CLI/插件。  
文件: orchestrate.py

10. FarsOrchestrator.record_result  
外部提交结果后，推进状态机。  
文件: orchestrate.py

11. cli.status / cli.init / cli_record  
你最常用的入口命令层，帮你触发 orchestrator。  
文件: cli.py

12. RunPodBackend.create_pod + wait_for_ready + run_remote  
GPU 执行三件套：开机、等可用、跑命令。  
文件: runpod_backend.py

给你一个“新手脑图”（一句话版）：
CLI 接收命令 -> Orchestrator 调 Lifecycle -> Lifecycle 出 Action -> 执行后写结果 -> StateMachine 算下一关 -> Workspace 持久化状态 -> 循环直到 done。

你现在最该做的不是读细节语法，而是每个函数只回答 3 件事：
1. 它读了什么输入  
2. 它写了什么输出  
3. 它把控制权交给了谁

如果你愿意，我下一条可以直接给你“第 1 次真读代码脚本”（按分钟安排：第几分钟看哪个函数、看完要回答哪一句话），你照着走一遍就能独立追调用链。

