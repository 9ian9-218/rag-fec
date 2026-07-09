#!/usr/bin/env python
"""测试远程 Embedding 和 Rerank API 连接。

用法：
    python scripts/test_remote_api.py
"""

import asyncio
import sys
from pathlib import Path

# 确保项目根目录在路径中
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings, apply_settings_to_environ


def test_embedding_api():
    """测试 Embedding API 配置和连接。"""
    print("=" * 60)
    print("测试 Embedding API")
    print("=" * 60)

    settings = get_settings()

    # 显示配置
    print(f"\nAPI 启用状态: {settings.embedding.api_enabled}")
    print(f"API 模型: {settings.embedding.api_model_name or '未设置'}")
    print(f"API Base URL: {settings.embedding.api_base_url or '未设置'}")
    print(f"API Key: {'已设置' if settings.embedding.api_key else '未设置'}")
    print(f"向量维度: {settings.embedding.dimension}")
    print(f"超时时间: {settings.embedding.api_timeout}s")

    if not settings.embedding.api_enabled:
        print("\n⚠️  Embedding API 未启用，跳过测试")
        print("如需启用，请在 .env 中设置 EMBEDDING_API_ENABLED=true")
        return False

    if not settings.embedding.api_key:
        print("\n❌ EMBEDDING_API_KEY 未设置")
        return False

    if not settings.embedding.api_base_url:
        print("\n❌ EMBEDDING_API_BASE_URL 未设置")
        return False

    if not settings.embedding.api_model_name:
        print("\n❌ EMBEDDING_API_MODEL_NAME 未设置")
        return False

    # 测试 API 连接
    print("\n正在测试 API 连接...")
    try:
        from src.storage.remote_embedding import build_remote_embedding_func

        embed_func = build_remote_embedding_func(settings)

        async def run_test():
            test_texts = ["测试文本1：FEC 前向纠错编码", "测试文本2：RAG 检索增强生成"]
            print(f"发送测试文本: {test_texts}")

            result = await embed_func(test_texts)
            print(f"✅ API 调用成功！")
            print(f"返回形状: {result.shape}")
            print(f"预期维度: ({len(test_texts)}, {settings.embedding.dimension})")
            print(f"向量示例（前5维）: {result[0][:5]}")

            # 验证维度
            if result.shape == (len(test_texts), settings.embedding.dimension):
                print("✅ 维度验证通过")
                return True
            else:
                print(f"❌ 维度不匹配！预期 {settings.embedding.dimension}，实际 {result.shape[1]}")
                return False

        return asyncio.run(run_test())

    except Exception as e:
        print(f"\n❌ API 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rerank_api():
    """测试 Rerank API 配置和连接。"""
    print("\n" + "=" * 60)
    print("测试 Rerank API")
    print("=" * 60)

    settings = get_settings()

    # 显示配置
    print(f"\nAPI 启用状态: {settings.models.rerank_api_enabled}")
    print(f"API 模型: {settings.models.rerank_api_model_name or '未设置'}")
    print(f"API Base URL: {settings.models.rerank_api_base_url or '未设置'}")
    print(f"API Key: {'已设置' if settings.models.rerank_api_key else '未设置'}")
    print(f"超时时间: {settings.models.rerank_api_timeout}s")

    if not settings.models.rerank_api_enabled:
        print("\n⚠️  Rerank API 未启用，跳过测试")
        print("如需启用，请在 .env 中设置 MODELS_RERANK_API_ENABLED=true")
        return False

    if not settings.models.rerank_api_key:
        print("\n❌ MODELS_RERANK_API_KEY 未设置")
        return False

    if not settings.models.rerank_api_base_url:
        print("\n❌ MODELS_RERANK_API_BASE_URL 未设置")
        return False

    if not settings.models.rerank_api_model_name:
        print("\n❌ MODELS_RERANK_API_MODEL_NAME 未设置")
        return False

    # 测试 API 连接
    print("\n正在测试 API 连接...")
    try:
        from src.storage.remote_rerank import build_remote_rerank_model_func

        rerank_func = build_remote_rerank_model_func(settings)

        if rerank_func is None:
            print("❌ Rerank API 初始化失败")
            return False

        async def run_test():
            test_query = "FEC 编码原理"
            test_docs = [
                "FEC（Forward Error Correction）是一种前向纠错技术，通过在发送端添加冗余数据，接收端可以检测并纠正传输错误。",
                "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。",
                "HTTP（HyperText Transfer Protocol）是互联网上应用最广泛的应用层协议。",
                "RAG（Retrieval-Augmented Generation）结合了检索和生成两种方法，提高了大语言模型的准确性。",
            ]

            print(f"查询: {test_query}")
            print(f"文档数量: {len(test_docs)}")

            result = await rerank_func(
                query=test_query,
                documents=test_docs,
                top_n=3
            )

            print(f"\n✅ API 调用成功！")
            print(f"返回结果数量: {len(result)}")
            print("\n排序结果:")
            for i, item in enumerate(result, 1):
                idx = item["index"]
                score = item["relevance_score"]
                doc_preview = test_docs[idx][:50] + "..."
                print(f"  {i}. [分数: {score:.4f}] {doc_preview}")

            return True

        return asyncio.run(run_test())

    except Exception as e:
        print(f"\n❌ API 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试。"""
    print("\n🔍 远程 API 连接测试")
    print("=" * 60)

    # 应用设置到环境变量
    apply_settings_to_environ()

    # 测试 Embedding API
    emb_success = test_embedding_api()

    # 测试 Rerank API
    rerank_success = test_rerank_api()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"Embedding API: {'✅ 通过' if emb_success else '❌ 失败/跳过'}")
    print(f"Rerank API:    {'✅ 通过' if rerank_success else '❌ 失败/跳过'}")

    if emb_success or rerank_success:
        print("\n🎉 至少一个 API 测试通过！")
        return 0
    else:
        print("\n⚠️  所有 API 测试均未通过，请检查配置")
        return 1


if __name__ == "__main__":
    sys.exit(main())
