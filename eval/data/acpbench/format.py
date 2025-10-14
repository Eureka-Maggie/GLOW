import json
import os
from tqdm import tqdm

def unify_answer_to_string():
    """
    读取指定的JSONL文件，并将其中'answer'字段的值统一转换为字符串格式。
    如果'answer'是列表，则转换为JSON字符串；否则，直接转换为字符串。
    """
    # --- 1. 文件路径配置 ---
    base_path = "/mnt/bn/maminghua-lf/projects/Code/eval/data/acpbench/"
    input_file_path = os.path.join(base_path, "acpbench_test.jsonl")
    
    # 定义一个新的输出文件名，以防覆盖原始文件
    output_file_path = os.path.join(base_path, "acpbench_test_fixed.jsonl")

    print("🚀 开始转换任务...")
    print(f"📂 输入文件: {input_file_path}")
    print(f"💾 输出文件: {output_file_path}")

    if not os.path.exists(input_file_path):
        print(f"❌ 错误: 输入文件未找到: {input_file_path}")
        return

    # --- 2. 逐行处理文件 ---
    try:
        num_lines = sum(1 for line in open(input_file_path, 'r', encoding='utf-8'))
        
        with open(input_file_path, 'r', encoding='utf-8') as f_in, \
             open(output_file_path, 'w', encoding='utf-8') as f_out:
            
            for line in tqdm(f_in, total=num_lines, desc="  转换中"):
                data = json.loads(line)

                # 检查 'answer' 字段是否存在
                if "answer" in data:
                    answer_value = data["answer"]
                    
                    # 如果 'answer' 是一个列表，则将其转换为JSON格式的字符串
                    if isinstance(answer_value, list):
                        data["answer"] = str(answer_value)
                    # 如果不是列表（例如整数、浮点数或已是字符串），也统一转换为字符串
                    else:
                        data["answer"] = str(answer_value)
                    print(data["answer"])
                
                # 将处理后的数据写回新文件
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        return

    # --- 3. 完成总结 ---
    print("\n--- ✨ 转换完成 ✨ ---")
    print("所有记录中的 'answer' 字段均已成功统一为字符串格式。")
    print(f"处理后的文件已保存至: {output_file_path}")
    print("现在你可以用这个新生成的文件来运行你的评估脚本了。")
    print("-------------------------")


if __name__ == '__main__':
    unify_answer_to_string()