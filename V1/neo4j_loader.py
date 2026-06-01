import json
import re
import sys
from pathlib import Path

from neo4j import GraphDatabase

# 定义函数编程的实体类型
ENTITY_TYPES = [
    "概念", "方法", "工具",
    "模型", "原则", "阶段",
    "角色", "工件", "技术",
    "框架", "模式", "流程",
    "标准", "实践", "语言"
]

# 定义函数编程的关系类型
RELATIONSHIP_TYPES = [
    "属于", "包含", "使用",
    "定义", "实现", "创建",
    "前置", "后置", "依赖",
    "派生", "应用于", "结合",
    "基于", "替代", "优化",
    "参与", "生成", "验证",
    "遵循", "扩展", "关联"
]

class Neo4jHandler:
    """Neo4j数据库操作处理器"""

    def __init__(self, uri, username, password):
        """初始化Neo4j处理器"""
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None

    def connect(self):
        """连接到Neo4j数据库"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            print("成功连接到Neo4j数据库")
        except Exception as e:
            print(f"连接Neo4j失败: {e}")
            raise

    def close(self):
        """关闭Neo4j连接"""
        if self.driver:
            self.driver.close()
            print("Neo4j连接已关闭")

    def clear_database(self):
        """从数据库中删除所有节点和关系"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                print("数据库已清空")
        except Exception as e:
            print(f"清空数据库时出错: {e}")

    def create_constraints(self):
        """为实体类型创建约束"""
        try:
            with self.driver.session() as session:
                # 为每个实体类型创建约束
                for entity_type in ENTITY_TYPES:
                    # 净化实体类型以适应Neo4j
                    clean_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', entity_type)
                    if not clean_type:  # 确保清理后的类型不为空
                        clean_type = "未知类型"
                    try:
                        # 创建约束（语法取决于Neo4j版本）
                        try:
                            # Neo4j 4.x+
                            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{clean_type}`) REQUIRE n.name IS UNIQUE")
                        except:
                            try:
                                # Neo4j 3.x
                                session.run(f"CREATE CONSTRAINT ON (n:`{clean_type}`) ASSERT n.name IS UNIQUE")
                            except:
                                print(f"无法为{clean_type}创建约束，跳过")
                    except Exception as e:
                        print(f"为{clean_type}创建约束时出错: {e}")
                print("已为所有实体类型创建约束")
        except Exception as e:
            print(f"设置约束时出错: {e}")

    def validate_triples(self, triples):
        """验证三元组数据的有效性，并修复可能导致错误的记录"""
        fixed_triples = []
        invalid_count = 0

        for triple in triples:
            try:
                # 确保所有必要的字段都存在
                if not all(key in triple for key in ["subject", "predicate", "object"]):
                    print(f"跳过缺少必要字段的三元组: {triple}")
                    invalid_count += 1
                    continue

                if not all(key in triple["subject"] for key in ["name", "type"]):
                    print(f"跳过主体缺少必要字段的三元组: {triple}")
                    invalid_count += 1
                    continue

                if not all(key in triple["object"] for key in ["name", "type"]):
                    print(f"跳过客体缺少必要字段的三元组: {triple}")
                    invalid_count += 1
                    continue

                # 确保类型字段在清理后不为空
                subject_type = triple["subject"]["type"]
                object_type = triple["object"]["type"]

                clean_subject_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', subject_type)
                clean_object_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', object_type)

                if not clean_subject_type:
                    print(f"修复主体类型为空的三元组: {triple}")
                    triple["subject"]["type"] = "未知类型"

                if not clean_object_type:
                    print(f"修复客体类型为空的三元组: {triple}")
                    triple["object"]["type"] = "未知类型"

                fixed_triples.append(triple)
            except Exception as e:
                print(f"验证三元组时出错: {e}, 三元组: {triple}")
                invalid_count += 1

        print(f"验证完成: 有效三元组 {len(fixed_triples)}, 无效/已修复三元组 {invalid_count}")
        return fixed_triples

    def add_triples_batch(self, triples, batch_size=100):
        """批量向数据库添加三元组"""
        # 首先验证并修复三元组
        validated_triples = self.validate_triples(triples)

        successful = 0
        total = len(validated_triples)

        for i in range(0, total, batch_size):
            batch = validated_triples[i:i+batch_size]
            try:
                with self.driver.session() as session:
                    with session.begin_transaction() as tx:
                        for triple in batch:
                            # 提取三元组信息
                            subject_name = triple["subject"]["name"]
                            subject_type = triple["subject"]["type"]
                            predicate = triple["predicate"]
                            object_name = triple["object"]["name"]
                            object_type = triple["object"]["type"]

                            # 净化类型以适应Neo4j
                            clean_subject_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', subject_type)
                            clean_object_type = re.sub(r'[^\w\u4e00-\u9fa5]', '', object_type)
                            clean_predicate = re.sub(r'[^\w\u4e00-\u9fa5]', '_', predicate)

                            # 确保类型不为空字符串
                            if not clean_subject_type:
                                clean_subject_type = "未知类型"
                            if not clean_object_type:
                                clean_object_type = "未知类型"
                            if not clean_predicate:
                                clean_predicate = "关联"

                            # 准备来源属性
                            source_props = ""
                            if "sources" in triple:
                                source_list = triple["sources"]
                                source_props = ", r.sources = $sources"
                            elif "source" in triple:
                                source_list = [triple["source"]]
                                source_props = ", r.source = $source"
                            else:
                                source_list = None

                            # 创建三元组
                            query = f"""
                            MERGE (s:`{clean_subject_type}` {{name: $subject_name}})
                            SET s.type = $subject_type
                            MERGE (o:`{clean_object_type}` {{name: $object_name}})
                            SET o.type = $object_type
                            MERGE (s)-[r:`{clean_predicate}`]->(o)
                            SET r.name = $predicate{source_props}
                            """

                            # 准备查询参数
                            params = {
                                "subject_name": subject_name,
                                "subject_type": subject_type,
                                "object_name": object_name,
                                "object_type": object_type,
                                "predicate": predicate
                            }

                            # 添加来源参数
                            if "sources" in triple:
                                params["sources"] = source_list
                            elif "source" in triple and source_list:
                                params["source"] = source_list[0]

                            tx.run(query, **params)

                        # 提交事务
                        tx.commit()
                        successful += len(batch)

                print(f"已添加 {i+len(batch)}/{total} 个三元组")

            except Exception as e:
                print(f"添加三元组批次 {i//batch_size + 1} 时出错: {e}")

        return successful

    def get_statistics(self):
        """获取知识图谱的统计信息"""
        try:
            stats = {}

            with self.driver.session() as session:
                # 获取节点数
                node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                stats["node_count"] = node_count

                # 获取关系数
                rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                stats["relationship_count"] = rel_count

                # 获取节点类型分布
                node_types_result = session.run(
                    "MATCH (n) RETURN DISTINCT labels(n)[0] as type, count(*) as count ORDER BY count DESC"
                )
                stats["node_types"] = [(record["type"], record["count"]) for record in node_types_result]

                # 获取关系类型分布
                rel_types_result = session.run(
                    "MATCH ()-[r]->() RETURN DISTINCT type(r) as type, count(*) as count ORDER BY count DESC"
                )
                stats["relationship_types"] = [(record["type"], record["count"]) for record in rel_types_result]

                return stats
        except Exception as e:
            print(f"获取统计信息时出错: {e}")
            return {"error": str(e)}


def load_triples_to_neo4j(triples_json_path, neo4j_uri, neo4j_username, neo4j_password,
                          clear_db=False, batch_size=100):
    """
    将三元组从JSON文件加载到Neo4j

    Args:
        triples_json_path: 包含三元组的JSON文件路径
        neo4j_uri: Neo4j数据库的URI
        neo4j_username: Neo4j用户名
        neo4j_password: Neo4j密码
        clear_db: 是否在加载三元组前清空数据库
        batch_size: 批处理大小

    Returns:
        包含已加载知识图谱统计信息的字典
    """
    # 检查文件是否存在
    triples_path = Path(triples_json_path)
    if not triples_path.exists():
        print(f"错误：文件不存在: {triples_json_path}")
        print(f"请确保文件位于: {triples_path.absolute()}")
        return {"error": f"文件不存在: {triples_json_path}"}

    # 从JSON加载三元组
    try:
        with open(triples_json_path, 'r', encoding='utf-8') as f:
            triples = json.load(f)

        print(f"✓ 已从 {triples_json_path} 加载 {len(triples)} 个三元组")
    except json.JSONDecodeError as e:
        print(f"✗ JSON 解析错误: {e}")
        return {"error": f"JSON 解析失败: {e}"}
    except Exception as e:
        print(f"✗ 从 {triples_json_path} 加载三元组时出错: {e}")
        return {"error": str(e)}

    # 初始化Neo4j处理器
    neo4j_handler = Neo4jHandler(neo4j_uri, neo4j_username, neo4j_password)

    try:
        neo4j_handler.connect()
    except Exception as e:
        print(f"✗ 无法连接到 Neo4j: {e}")
        print(f"  请检查:")
        print(f"  1. Neo4j 服务是否正在运行")
        print(f"  2. URI 是否正确: {neo4j_uri}")
        print(f"  3. 用户名和密码是否正确")
        return {"error": f"Neo4j 连接失败: {e}"}

    try:
        # 如果请求，清空数据库
        if clear_db:
            neo4j_handler.clear_database()

        # 创建约束
        neo4j_handler.create_constraints()

        # 将三元组添加到数据库
        print("⏳ 正在将三元组批量添加到Neo4j...")
        successful_triples = neo4j_handler.add_triples_batch(triples, batch_size=batch_size)

        # 获取统计信息
        stats = neo4j_handler.get_statistics()
        stats["successful_triples"] = successful_triples
        stats["total_triples"] = len(triples)

        print(f"\n✓ 成功添加 {successful_triples}/{len(triples)} 个三元组到Neo4j")
        print(f"✓ Neo4j 现在包含 {stats['node_count']} 个节点和 {stats['relationship_count']} 个关系")

        return stats

    except Exception as e:
        print(f"✗ 加载过程中出错: {e}")
        return {"error": str(e)}
    finally:
        # 关闭Neo4j连接
        neo4j_handler.close()


if __name__ == "__main__":
    """命令行入口 - 支持参数化配置"""

    # 默认配置
    default_config = {
        "triples_json_path": "functional_programming_kg_results/functional_programming_triples_unique.json",
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "neo4j123",
        "clear_db": False,
        "batch_size": 100
    }

    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print("用法: python neo4j_loader.py [options]")
            print("\n选项:")
            print("  -f, --file PATH              指定三元组 JSON 文件路径")
            print("  -u, --uri URI                指定 Neo4j URI (默认: bolt://localhost:7687)")
            print("  -U, --username USER          指定 Neo4j 用户名 (默认: neo4j)")
            print("  -p, --password PASS          指定 Neo4j 密码 (默认: neo4j123)")
            print("  --no-clear                   不清空数据库")
            print("  -b, --batch-size SIZE        指定批处理大小 (默认: 100)")
            print("  -h, --help                   显示此帮助信息")
            print("\n示例:")
            print("  python neo4j_loader.py -f ./my_triples.json -u bolt://localhost:7688")
            sys.exit(0)

        # 处理参数
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]

            if arg in ["-f", "--file"] and i + 1 < len(sys.argv):
                default_config["triples_json_path"] = sys.argv[i + 1]
                i += 2
            elif arg in ["-u", "--uri"] and i + 1 < len(sys.argv):
                default_config["neo4j_uri"] = sys.argv[i + 1]
                i += 2
            elif arg in ["-U", "--username"] and i + 1 < len(sys.argv):
                default_config["neo4j_username"] = sys.argv[i + 1]
                i += 2
            elif arg in ["-p", "--password"] and i + 1 < len(sys.argv):
                default_config["neo4j_password"] = sys.argv[i + 1]
                i += 2
            elif arg == "--no-clear":
                default_config["clear_db"] = False
                i += 1
            elif arg in ["-b", "--batch-size"] and i + 1 < len(sys.argv):
                try:
                    default_config["batch_size"] = int(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    print(f"错误: 批处理大小必须是整数")
                    sys.exit(1)
            else:
                i += 1

    # 打印配置信息
    print("=" * 60)
    print("Neo4j 三元组加载器")
    print("=" * 60)
    print(f"\n📋 配置信息:")
    print(f"  文件路径: {default_config['triples_json_path']}")
    print(f"  Neo4j URI: {default_config['neo4j_uri']}")
    print(f"  用户名: {default_config['neo4j_username']}")
    print(f"  清空数据库: {default_config['clear_db']}")
    print(f"  批处理大小: {default_config['batch_size']}")
    print()

    # 加载三元组到Neo4j
    stats = load_triples_to_neo4j(
        triples_json_path=default_config['triples_json_path'],
        neo4j_uri=default_config['neo4j_uri'],
        neo4j_username=default_config['neo4j_username'],
        neo4j_password=default_config['neo4j_password'],
        clear_db=default_config['clear_db'],
        batch_size=default_config['batch_size']
    )

    # 打印统计信息
    if "error" not in stats:
        print("\n" + "=" * 60)
        print("📊 知识图谱统计信息")
        print("=" * 60)
        print(f"总节点数: {stats['node_count']}")
        print(f"总关系数: {stats['relationship_count']}")
        print(f"成功加载: {stats['successful_triples']}/{stats['total_triples']} 个三元组")

        if stats.get('node_types'):
            print(f"\n节点类型分布:")
            for node_type, count in stats['node_types'][:10]:  # 只显示前10个
                print(f"  {node_type}: {count}")

        if stats.get('relationship_types'):
            print(f"\n关系类型分布:")
            for rel_type, count in stats['relationship_types'][:10]:  # 只显示前10个
                print(f"  {rel_type}: {count}")

        print("\n✓ 加载完成！")
    else:
        print(f"\n✗ 加载失败: {stats['error']}")