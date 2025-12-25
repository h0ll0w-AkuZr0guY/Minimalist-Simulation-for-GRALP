# 贡献指南文档

## 1. 贡献概述

欢迎您为 Minimalist-Simulation-for-GRALP 项目做贡献！无论您是经验丰富的开发者还是刚入门的小白，我们都非常欢迎您的参与。贡献可以是代码改进、文档更新、bug 修复、新功能开发等多种形式。

### 1.1 为什么贡献？

- 提高您的编程技能和协作能力
- 深入了解强化学习和机器人仿真技术
- 与社区共同打造更好的开源项目
- 获得项目维护者和社区的认可
- 为您的简历增添亮点

### 1.2 贡献的形式

- **代码贡献**：修复 bug、添加新功能、优化性能
- **文档贡献**：更新文档、添加示例、翻译内容
- **测试贡献**：编写测试用例、改进测试覆盖率
- **问题报告**：报告 bug、提出改进建议
- **代码审查**：参与代码审查，提供反馈

## 2. 贡献流程

### 2.1 准备工作

1. **了解项目**：阅读项目文档，了解项目的结构和功能
2. **安装开发环境**：按照项目 README 安装依赖和设置开发环境
3. **熟悉代码规范**：阅读本指南中的代码规范部分
4. **加入社区**：关注项目动态，参与讨论

### 2.2 寻找贡献机会

- 查看项目的 Issues 列表，寻找适合自己的任务
- 查找带有 "good first issue" 标签的问题，这些问题适合初学者
- 提出自己的改进建议或新功能想法
- 修复文档中的错误或不完善之处

### 2.3 贡献步骤

1. **Fork 仓库**：点击 GitHub 上的 Fork 按钮，创建自己的仓库副本
2. **克隆仓库**：将 Fork 的仓库克隆到本地
   ```bash
   git clone https://github.com/your-username/Minimalist-Simulation-for-GRALP.git
   cd Minimalist-Simulation-for-GRALP
   ```
3. **创建分支**：为您的贡献创建一个新分支
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **编写代码**：按照代码规范编写代码，实现功能或修复 bug
5. **运行测试**：确保您的代码通过所有测试
   ```bash
   python -m pytest tests/
   ```
6. **提交代码**：提交您的更改，遵循提交规范
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```
7. **推送代码**：将您的分支推送到 GitHub
   ```bash
   git push origin feature/your-feature-name
   ```
8. **创建 Pull Request**：在 GitHub 上创建 Pull Request，描述您的贡献

### 2.4 Pull Request 流程

1. **提交 Pull Request**：填写详细的描述，说明您的更改内容和目的
2. **代码审查**：项目维护者和其他贡献者会审查您的代码
3. **修改代码**：根据审查反馈，修改您的代码
4. **通过测试**：确保所有测试通过
5. **合并代码**：代码审查通过后，项目维护者会合并您的代码

## 3. 代码规范

### 3.1 语言和框架

- 使用 Python 3.8+ 编写代码
- 遵循 PEP 8 代码风格指南
- 使用类型注解，提高代码可读性和类型安全性
- 使用 docstring 为函数和类添加文档

### 3.2 命名规范

- **变量名**：使用小写字母和下划线（snake_case）
- **函数名**：使用小写字母和下划线（snake_case）
- **类名**：使用驼峰命名法（CamelCase）
- **常量名**：使用大写字母和下划线（UPPER_SNAKE_CASE）
- **模块名**：使用小写字母和下划线（snake_case）

### 3.3 代码结构

- 每个文件只包含一个主要的类或功能
- 使用清晰的模块划分，避免过长的文件
- 函数和方法的长度不宜过长，建议不超过 50 行
- 使用注释解释复杂的逻辑和算法

### 3.4 代码示例

```python
# 正确的代码示例
class RobotController:
    """机器人控制器类，负责机器人的运动控制
    
    属性:
        max_linear_velocity: 最大线速度
        max_angular_velocity: 最大角速度
    """
    
    def __init__(self, max_linear_velocity: float = 0.5, max_angular_velocity: float = 1.0):
        """初始化机器人控制器
        
        参数:
            max_linear_velocity: 最大线速度
            max_angular_velocity: 最大角速度
        """
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
    
    def apply_control(self, linear_vel: float, angular_vel: float) -> None:
        """应用控制命令
        
        参数:
            linear_vel: 期望的线速度
            angular_vel: 期望的角速度
        """
        # 限制速度在安全范围内
        linear_vel = self._clamp(linear_vel, -self.max_linear_velocity, self.max_linear_velocity)
        angular_vel = self._clamp(angular_vel, -self.max_angular_velocity, self.max_angular_velocity)
        
        # 应用控制命令
        # ...
    
    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """将值限制在指定范围内
        
        参数:
            value: 要限制的值
            min_val: 最小值
            max_val: 最大值
        
        返回:
            限制后的值
        """
        return max(min_val, min(value, max_val))
```

### 3.5 代码格式化

- 使用 Black 工具格式化代码
  ```bash
  black .
  ```
- 使用 Flake8 检查代码风格
  ```bash
  flake8 .
  ```
- 使用 MyPy 检查类型注解
  ```bash
  mypy .
  ```

## 4. 测试要求

### 4.1 测试的重要性

- 确保代码的正确性和稳定性
- 防止引入新的 bug
- 提高代码的可维护性
- 验证新功能的正常工作

### 4.2 测试类型

- **单元测试**：测试单个函数或类的功能
- **集成测试**：测试多个模块之间的交互
- **性能测试**：测试代码的性能和效率

### 4.3 编写测试用例

- 使用 pytest 框架编写测试用例
- 为每个函数和类编写至少一个测试用例
- 测试用例应覆盖正常情况和边界情况
- 测试用例应具有清晰的名称和描述

### 4.4 测试示例

```python
# tests/test_robot_controller.py
import pytest
from robot import RobotController

class TestRobotController:
    """测试机器人控制器类"""
    
    def test_init(self):
        """测试初始化功能"""
        controller = RobotController(0.5, 1.0)
        assert controller.max_linear_velocity == 0.5
        assert controller.max_angular_velocity == 1.0
    
    def test_apply_control(self):
        """测试应用控制命令"""
        controller = RobotController(0.5, 1.0)
        # 测试正常速度
        controller.apply_control(0.3, 0.5)
        # 测试超过最大速度
        controller.apply_control(1.0, 2.0)
        # 测试负速度
        controller.apply_control(-0.3, -0.5)
    
    def test_clamp(self):
        """测试速度限制功能"""
        controller = RobotController(0.5, 1.0)
        # 测试正常情况
        assert controller._clamp(0.3, -0.5, 0.5) == 0.3
        # 测试超过最大值
        assert controller._clamp(1.0, -0.5, 0.5) == 0.5
        # 测试低于最小值
        assert controller._clamp(-1.0, -0.5, 0.5) == -0.5
```

### 4.5 运行测试

- 运行所有测试
  ```bash
  python -m pytest
  ```
- 运行特定测试文件
  ```bash
  python -m pytest tests/test_robot.py
  ```
- 运行特定测试用例
  ```bash
  python -m pytest tests/test_robot.py::TestRobot::test_init
  ```
- 查看测试覆盖率
  ```bash
  python -m pytest --cov=.
  ```

## 5. 提交规范

### 5.1 提交信息格式

提交信息应遵循以下格式：

```
<类型>: <简短描述>

<详细描述（可选）>

<关联的 Issue 编号（可选）>
```

### 5.2 提交类型

| 类型 | 描述 |
|------|------|
| feat | 新功能 |
| fix | 修复 bug |
| docs | 文档更新 |
| style | 代码风格更改 |
| refactor | 代码重构 |
| perf | 性能优化 |
| test | 测试相关 |
| build | 构建相关 |
| ci | CI 配置更改 |
| chore | 其他更改 |

### 5.3 提交示例

```
feat: 添加机器人碰撞检测功能

实现了机器人与障碍物之间的碰撞检测功能，包括：
1. 基于 PyBullet 的碰撞检测接口
2. 碰撞事件的处理逻辑
3. 碰撞奖励的计算

fixes #123
```

```
fix: 修复激光雷达数据可视化错误

修复了激光雷达数据可视化时，角度计算错误导致的可视化偏移问题

fixes #456
```

```
docs: 更新 README 中的安装说明

更新了项目 README 中的依赖安装说明，添加了 Python 版本要求和虚拟环境设置指南
```

## 6. 文档贡献

### 6.1 文档的重要性

- 帮助新用户快速上手
- 提高项目的可理解性
- 展示项目的功能和价值
- 吸引更多贡献者

### 6.2 文档类型

- **项目文档**：README、安装指南、快速入门
- **API 文档**：函数和类的文档字符串
- **教程文档**：分步教程、示例代码
- **架构文档**：项目架构、模块依赖
- **贡献指南**：本指南

### 6.3 文档编写规范

- 使用 Markdown 格式编写文档
- 保持文档的清晰和简洁
- 使用示例代码，帮助用户理解
- 定期更新文档，保持与代码同步
- 使用一致的术语和风格

### 6.4 文档测试

- 确保文档中的示例代码可以正常运行
- 验证文档中的命令和步骤可以正常执行
- 检查文档中的链接是否有效
- 测试文档的可读性和可理解性

## 7. 问题报告和建议

### 7.1 报告 bug

如果您发现了 bug，请按照以下步骤报告：

1. **搜索现有 Issues**：检查是否已经有人报告了相同的 bug
2. **创建新 Issue**：如果没有，创建一个新的 Issue
3. **提供详细信息**：
   - 清晰的 bug 描述
   - 重现步骤
   - 预期行为和实际行为
   - 环境信息（Python 版本、操作系统等）
   - 错误信息和日志
   - 可能的解决方案（可选）

### 7.2 提出建议

如果您有改进建议或新功能想法，请：

1. **创建新 Issue**：描述您的建议或想法
2. **提供详细说明**：
   - 建议的功能或改进
   - 为什么需要这个功能
   - 预期的效果
   - 可能的实现方案（可选）
3. **参与讨论**：与项目维护者和其他贡献者讨论您的建议

## 8. 代码审查

### 8.1 代码审查的重要性

- 提高代码质量
- 确保代码符合规范
- 分享知识和经验
- 发现潜在的问题
- 培养团队协作精神

### 8.2 参与代码审查

1. **关注 Pull Requests**：定期查看项目的 Pull Requests
2. **提供有建设性的反馈**：
   - 指出代码中的问题和改进点
   - 提供具体的建议
   - 保持礼貌和尊重
   - 解释反馈的原因
3. **回应他人的反馈**：
   - 认真考虑他人的反馈
   - 解释您的设计决策
   - 必要时修改代码
   - 感谢他人的反馈

### 8.3 代码审查的关注点

- **代码正确性**：代码是否实现了预期功能
- **代码质量**：代码是否清晰、简洁、可维护
- **代码规范**：代码是否符合项目的代码规范
- **性能**：代码是否高效，有没有性能问题
- **测试覆盖率**：是否有足够的测试用例
- **文档**：是否有完整的文档

## 9. 贡献者行为准则

### 9.1 基本准则

- **尊重**：尊重所有贡献者，无论经验水平如何
- **包容**：欢迎不同背景和观点的贡献者
- **合作**：与其他贡献者合作，共同解决问题
- **负责**：对自己的贡献负责，确保质量
- **学习**：不断学习和提高自己的技能

### 9.2 禁止行为

- 人身攻击或侮辱性语言
- 歧视或偏见言论
- 不尊重他人的劳动成果
- 恶意破坏或提交无效代码
- 未经授权的代码提交

### 9.3 冲突解决

- 保持冷静和理性
- 尝试理解对方的观点
- 寻求第三方的帮助和调解
- 遵循项目维护者的决定

## 10. 致谢

感谢所有为 Minimalist-Simulation-for-GRALP 项目做出贡献的开发者！您的贡献对于项目的成功至关重要。

### 10.1 贡献者名单

我们会在项目的 README 文件中列出所有贡献者的名字，以表达我们的感谢和认可。

### 10.2 获得认可

- 您的贡献将被记录在项目的 Git 历史中
- 您将成为项目的贡献者，出现在 GitHub 上的贡献者列表中
- 优秀的贡献者可能会被邀请成为项目的维护者

## 11. 总结

贡献开源项目是一个学习和成长的过程，我们欢迎所有愿意参与的开发者。通过遵循本指南，您可以顺利地为项目做出贡献，并获得社区的认可。

如果您有任何疑问或需要帮助，请随时在项目的 Issues 中提问，或联系项目维护者。

再次感谢您的关注和贡献！

## 12. 联系信息

- **项目 GitHub 仓库**：[https://github.com/your-username/Minimalist-Simulation-for-GRALP](https://github.com/your-username/Minimalist-Simulation-for-GRALP)
- **Issue 跟踪**：[https://github.com/your-username/Minimalist-Simulation-for-GRALP/issues](https://github.com/your-username/Minimalist-Simulation-for-GRALP/issues)
- **讨论区**：[GitHub Discussions](https://github.com/your-username/Minimalist-Simulation-for-GRALP/discussions)

祝您贡献愉快！