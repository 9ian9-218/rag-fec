整体 Schema 参考结构
1. Node 节点类型
Document：整个 MD 文档
Chapter：章→节→小节树（level、sectionId、title 等）
Concept：专业概念、术语、定义
Entity：所有「具体对象」：码族、算法、数学对象、信道、插图文件… 都用这一类，靠 entityKind 区分
Attribute：属性、参数 / 特征（常通过 HAS_ATTRIBUTE 挂在 Entity 上）
Case：案例、示例、场景
2. Relationship 关系类型
包含(contain)：文档→章节、章节→小节、章节→实体
定义(define)：章节 / 文本→定义某个概念
组成(compose)：大组件→子组件
依赖(depend)：A 依赖 B
关联(relate)：概念与概念相互关联
拥有(has_attr)：实体→拥有某属性
属于(belong)：子条目→归属父分类

# FEC 图谱节点字段表

口径：`docs/schema.md`、`docs/FEC知识图谱建图手册.md` §3–§4。Neo4j 关系名：`CONTAINS`、`DEFINES`、`COMPOSES`、`DEPENDS_ON`、`RELATES_TO`、`HAS_ATTRIBUTE`、`BELONGS_TO`。

## 全局（所有节点）

| 字段 | 必填 | 说明 |
|------|:----:|------|
| `nodeId` | 是 | 全局唯一字符串主键 |
| `labels` | 是 | Neo4j 标签：`Document` \| `Chapter` \| `Concept` \| `Entity` \| `Attribute` \| `Case` |
| `name` | 是* | 展示名（* `Document` 常用 `title` 作主显名，宽表可并存） |
| `preferredTerm` / `fsn` | 否 | 全称 / 英文规范名 |
| `synonyms` | 否 | 别名（JSON 或分号分隔） |
| `conceptType` | 否 | 大类枚举（兼容宽表） |
| `description` | 否 | 短摘要 |
| `textForEmbedding` | 否 | 拼接文本，供向量导出 / 检索 |
| `sourceRef` | 否 | 溯源（如 `book:…` + `sec:…`） |
| `milvusPrimaryKey` | 否 | 与 Milvus 行主键 `id` 对齐 |

## 按 Label

### Document

| 字段 | 必填 | 说明 |
|------|:----:|------|
| `uri` | 推荐 | 文档路径或标识 |
| `title` | 推荐 | 书名 / 文档标题 |
| `language` | 否 | 语言 |
| `authors` | 否 | 作者 |
| `bibKey` | 否 | 书目键 |

### Chapter

| 字段 | 必填 | 说明 |
|------|:----:|------|
| `level` | 推荐 | 1=章，2=节… |
| `chapterNo` | 否 | 章号 |
| `sectionId` | 推荐 | 节编号 / 锚点 |
| `title` | 推荐 | 章节标题 |

### Concept

| 字段 | 必填 | 说明 |
|------|:----:|------|
| `aliases` | 否 | 术语别名 |
| `definitionSnippet` | 推荐 | 定义摘录 |

### Entity

| 字段 | 必填 | 说明 |
|------|:----:|------|
| `entityKind` | 是 | 见下表 |
| `schemeKind` | 否 | 体制细分（如 ldpc / block） |
| `algorithmRole` | 否 | encode / decode / both 等 |
| `mathKind` | 否 | 数学对象类型 |
| `parameterSummary` | 否 | 参数摘要 |
| `hardSoft` / `iterative` | 否 | 算法特征 |
| `uri` / `relativePath` | media 必填 | 资源路径 |
| `sha256` | media 推荐 | 文件哈希去重 |
| `caption` / `ocrText` | media 推荐 | 图题 / OCR |
| `linkedChunkId` | 否 | 绑定文本分块 id |
| `embeddingModel` | 否 | 图像向量模型版本 |

#### entityKind 枚举（FEC）

| entityKind | 含义 |
|------------|------|
| `coding_paradigm` | 顶层体制 |
| `coding_scheme` | 码族 |
| `algorithm` | 编/译码算法 |
| `math_object` | GF、矩阵、图等 |
| `channel_model` | 信道模型 |
| `code_instance` | (n,k,t) 等实例 |
| `person` | 人物 |
| `tool` | 工具 / 接口 |
| `media` | 插图 / 示意图文件 |
| `module` | 模块 / 组件 |

### Attribute

| 字段 | 必填 | 说明 |
|------|:----:|------|
| `value` | 否 | 属性值 |
| `unit` | 否 | 单位 |

### Case

| 字段 | 必填 | 说明 |
|------|:----:|------|
| （无额外必填） | — | 主要用 `name`、`description`、`sourceRef` |

## 关系边（可选属性）

| 字段 | 说明 |
|------|------|
| `relationshipId` | 边全局唯一 id |
| `weight` / `condition` / `metric` | 加权或条件 |
| `sourceRef` / `order` | 溯源、顺序 |
