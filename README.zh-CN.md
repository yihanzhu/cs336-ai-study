# CS336 AI 陪学

[English](README.md) · [双语页面](outputs/CS336-lesson-plans.html) · [开始陪学](outputs/开始陪学.txt)

299点英文教案与299点中文教案共用编号和来源。页面默认英文，陪学按学习者自己选择的语言。AI每轮解释一个点，走一个例子，问一个理解问题，等回答后继续。无需先看视频或完成大作业。

## 开始

1. 下载仓库，让能读文件的agent打开目录。
2. 运行 `python3 work/init_learning.py --language zh`，从空白模板建立本地进度。英文用 `--language en`。已有进度不会被覆盖。
3. 告诉agent：“先读AGENTS.md、outputs/开始陪学.txt和local/learning-progress.md，按我的语言偏好继续。每轮一个理解问题，等我回答。”

没有Python时，把空白模板复制为 `local/learning-progress.md`。没有文件工具时，向agent提供当前教案和进度文本，并自行保存更新结果。不限于coding agent；CLAUDE.md导入同一套规则，其他agent可以按提示直接读取。

## 个人记录只留本地

| 可公开 | 仅本地 |
|---|---|
| 教案、来源、空白模板 | local/learning-progress.md |
| Git与发布资料包 | 被Git忽略，不进入发布包 |
| 课程更新可以拉取 | 个人进度不自动提交或上传 |

新下载的仓库没有任何人的学习记录。换电脑时，另外携带本地进度文件，或明确选择独立私有存储。仓库不提供账号、进度服务器、遥测或自动云同步。agent平台如何保存聊天、是否调用云端模型，由所用工具决定；“本地进度”不等于“本地运行模型”。

## 内容与边界

完整范围是原目录299点的教学设计。五场嘉宾完整课堂未取得；Dan Fu有录像字幕，但本场独立幻灯片未取得。缺口不编造。原材料核验基线仍为1.2，不声称人工逐句听校、逐帧审片、GPU训练复跑或全量独立审校。

299点教案和逐点限制均有英中两版；历史详细备课与核验笔记保留中文。编号、哈希和脚本检查不能替代所有语义判断。

项目原创代码与教学文字采用MIT许可。上游课程、论文、数据及代码保留各自权利和许可；见LICENSE与THIRD_PARTY_NOTICES.md。官方PDF、完整字幕、音视频和缓存不随仓库分发。本项目不是Stanford官方课程、官方翻译或认证。

## 构建

仅使用已跟踪文件即可重建教案，无需原始缓存或旧ZIP：

```sh
python3 work/build_bilingual.py
python3 work/verify_lesson_plans.py
python3 work/verify_bilingual.py
python3 work/verify_publication.py
```

全量来源重建 `python3 work/build_release.py` 仍依赖本机原始材料。只读教案无需安装GPU环境。
