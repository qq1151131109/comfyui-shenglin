#!/usr/bin/env python3
"""
RSS Tool 节点测试脚本
测试所有节点是否能正常导入和初始化
"""

import sys
import traceback

def test_node_imports():
    """测试所有节点的导入"""
    test_results = {}
    
    nodes_to_test = [
        ("MultiRSSCollector", "MultiRSSCollector"),
        ("BatchArticleExtractor", "BatchArticleExtractor"), 
        ("ContentDeduplicator", "ContentDeduplicator"),
        ("TopicClusteringEngine", "TopicClusteringEngine"),
        ("TopicReportGenerator", "TopicReportGenerator"),
        ("ContentExporter", "ContentExporter"),
    ]
    
    print("🧪 开始测试RSS工具节点...")
    print("=" * 50)
    
    for module_name, class_name in nodes_to_test:
        try:
            print(f"测试 {class_name}...", end=" ")
            
            # Import the module
            module = __import__(module_name, fromlist=[class_name])
            node_class = getattr(module, class_name)
            
            # Try to instantiate
            instance = node_class()
            
            # Check required methods
            if hasattr(instance, 'INPUT_TYPES') and hasattr(instance, 'execute'):
                print("✅ 通过")
                test_results[class_name] = "success"
            else:
                print("⚠️  缺少必要方法")
                test_results[class_name] = "warning"
                
        except ImportError as e:
            print(f"❌ 导入失败: {e}")
            test_results[class_name] = f"import_error: {e}"
        except Exception as e:
            print(f"❌ 实例化失败: {e}")
            test_results[class_name] = f"init_error: {e}"
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    success_count = sum(1 for result in test_results.values() if result == "success")
    total_count = len(test_results)
    
    print(f"总节点数: {total_count}")
    print(f"成功: {success_count}")
    print(f"失败: {total_count - success_count}")
    
    if success_count == total_count:
        print("🎉 所有节点测试通过!")
        return True
    else:
        print("\n❌ 失败详情:")
        for node, result in test_results.items():
            if result != "success":
                print(f"- {node}: {result}")
        return False

def test_dependencies():
    """测试依赖库"""
    print("\n🔍 检查依赖库...")
    
    dependencies = [
        "feedparser",
        "newspaper",
        "nltk", 
        "sklearn",
        "jieba",
        "numpy",
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - 未安装")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠️  缺少依赖: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False
    
    print("🎉 所有依赖已安装!")
    return True

if __name__ == "__main__":
    print("ComfyUI RSS Tool 测试脚本")
    print("=" * 50)
    
    deps_ok = test_dependencies()
    nodes_ok = test_node_imports()
    
    if deps_ok and nodes_ok:
        print("\n🎉 所有测试通过! RSS工具可以正常使用。")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查上述错误信息。")
        sys.exit(1)