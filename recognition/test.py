from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from dataset import create_dataset
import torch
import logging
import random

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RadiologyTester:
    """放射学模型测试器 - 使用真实训练数据"""
    
    def __init__(self, adapter_path="./radiology-simplifier-output"):
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.test_dataset = None
        self._load_model_and_data()
    
    def _load_model_and_data(self):
        """加载模型和测试数据"""
        logger.info("🚀 加载模型和测试数据...")
        
        try:
            # 加载基础模型
            base_model_name = "google/flan-t5-base"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            
            # 加载适配器
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            
            # 加载测试数据集
            dataset_processor = create_dataset(base_model_name)
            tokenized_datasets = dataset_processor.get_tokenized_datasets()
            self.test_dataset = tokenized_datasets["test"]
            
            # 移动到GPU（如果可用）
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                logger.info("✅ 模型已加载到GPU")
            else:
                logger.info("✅ 模型已加载到CPU")
                
            logger.info(f"🎯 测试集加载完成: {len(self.test_dataset)} 个样本")
            
        except Exception as e:
            logger.error(f"❌ 加载失败: {e}")
            raise
    
    def decode_original_text(self, tokenized_sample):
        """从tokenized数据解码原始文本"""
        input_ids = tokenized_sample["input_ids"]
        labels = tokenized_sample["labels"]
        
        # 解码输入文本
        original_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # 解码标签（处理-100）
        labels = [token for token in labels if token != -100]
        expected_text = self.tokenizer.decode(labels, skip_special_tokens=True)
        
        return original_text, expected_text
    
    def generate_explanation(self, medical_report):
        """生成放射学报告解释"""
        # 最佳提示词
        prompt = "Medical report: {}\nSimple explanation:"
        test_text = prompt.format(medical_report)
        
        inputs = self.tokenizer(
            test_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def test_specific_samples(self, sample_indices):
        """测试特定样本"""
        print(f"\n🎯 测试特定样本: {sample_indices}")
        print("=" * 70)
        
        for idx in sample_indices:
            if idx >= len(self.test_dataset):
                print(f"❌ 样本索引 {idx} 超出范围")
                continue
                
            sample = self.test_dataset[idx]
            original_text, expected_text = self.decode_original_text(sample)
            
            print(f"\n📋 样本 {idx}:")
            print(f"🧬 原始报告: {original_text[:200]}..." if len(original_text) > 200 else f"🧬 原始报告: {original_text}")
            print(f"🎯 期望输出: {expected_text}" if expected_text.strip() else "🎯 期望输出: [无标签]")
            
            # 生成解释
            explanation = self.generate_explanation(original_text)
            print(f"💡 模型输出: {explanation}")
            
            # 质量评估
            self._evaluate_output(explanation, expected_text)
    
    def test_random_samples(self, num_samples=10):
        """随机测试样本"""
        print(f"\n🎲 随机测试 {num_samples} 个样本")
        print("=" * 70)
        
        random_indices = random.sample(range(len(self.test_dataset)), num_samples)
        
        for i, idx in enumerate(random_indices, 1):
            sample = self.test_dataset[idx]
            original_text, expected_text = self.decode_original_text(sample)
            
            print(f"\n🔍 随机样本 {i} (索引: {idx}):")
            print(f"🧬 原始报告: {original_text[:150]}..." if len(original_text) > 150 else f"🧬 原始报告: {original_text}")
            
            # 生成解释
            explanation = self.generate_explanation(original_text)
            print(f"💡 模型输出: {explanation}")
            
            # 简单质量检查
            if len(explanation) > 20 and any(term in explanation.lower() for term in ['lung', 'heart', 'chest', 'tube', 'x-ray']):
                print("✅ 输出质量: 良好")
            else:
                print("⚠️  输出质量: 需要检查")
    
    def test_by_length(self, min_len=50, max_len=200, num_samples=5):
        """按长度筛选测试样本"""
        print(f"\n📏 测试长度在 {min_len}-{max_len} 字符的样本")
        print("=" * 70)
        
        suitable_samples = []
        for i in range(min(1000, len(self.test_dataset))):  # 检查前1000个样本
            sample = self.test_dataset[i]
            original_text, _ = self.decode_original_text(sample)
            
            if min_len <= len(original_text) <= max_len:
                suitable_samples.append((i, original_text))
            
            if len(suitable_samples) >= num_samples:
                break
        
        for idx, (sample_idx, original_text) in enumerate(suitable_samples, 1):
            print(f"\n📋 适中长度样本 {idx}:")
            print(f"🧬 原始报告: {original_text}")
            
            explanation = self.generate_explanation(original_text)
            print(f"💡 模型输出: {explanation}")
    
    def test_medical_categories(self):
        """按医学类别测试"""
        print(f"\n🏥 按医学类别测试")
        print("=" * 70)
        
        # 医学关键词分类
        categories = {
            "心脏相关": ["cardiomegaly", "heart", "cardiac", "aortic"],
            "肺部相关": ["pulmonary", "lung", "pneumonia", "infiltrate"],
            "积液相关": ["effusion", "pleural", "fluid"],
            "骨骼相关": ["spine", "vertebral", "degenerative", "osteophyte"]
        }
        
        for category, keywords in categories.items():
            print(f"\n🔍 {category}:")
            found = False
            
            for i in range(min(500, len(self.test_dataset))):  # 搜索前500个样本
                sample = self.test_dataset[i]
                original_text, _ = self.decode_original_text(sample)
                
                # 检查是否包含该类别的关键词
                if any(keyword in original_text.lower() for keyword in keywords):
                    found = True
                    print(f"📋 样本 {i}: {original_text[:100]}...")
                    
                    explanation = self.generate_explanation(original_text)
                    print(f"💡 简化: {explanation}")
                    print("-" * 50)
                    break
            
            if not found:
                print(f"  未找到{category}的样本")
    
    def _evaluate_output(self, model_output, expected_output):
        """评估输出质量"""
        if not expected_output.strip():
            print("⚠️  质量评估: 无参考标签")
            return
        
        # 简单相似度检查
        model_words = set(model_output.lower().split())
        expected_words = set(expected_output.lower().split())
        common_words = model_words & expected_words
        
        similarity = len(common_words) / max(len(expected_words), 1)
        
        if similarity > 0.3:
            print(f"✅ 质量评估: 高相似度 ({similarity:.2f})")
        elif similarity > 0.1:
            print(f"⚠️  质量评估: 中等相似度 ({similarity:.2f})")
        else:
            print(f"❌ 质量评估: 低相似度 ({similarity:.2f})")
    
    def comprehensive_test(self):
        """综合测试"""
        print("🎯 放射学模型综合测试报告")
        print("=" * 70)
        
        # 1. 测试前5个样本
        self.test_specific_samples([0, 1, 2, 3, 4])
        
        # 2. 随机测试
        self.test_random_samples(5)
        
        # 3. 按长度测试
        self.test_by_length()
        
        # 4. 按类别测试
        self.test_medical_categories()
        
        print("\n🎉 综合测试完成!")

def main():
    """主测试函数"""
    try:
        # 初始化测试器
        tester = RadiologyTester("./radiology-simplifier-output")
        
        # 运行综合测试
        tester.comprehensive_test()
        
        # 交互式测试特定样本
        print("\n🎯 交互式样本测试")
        print("输入样本索引进行测试，输入 'quit' 退出")
        
        while True:
            try:
                user_input = input("\n请输入样本索引: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.isdigit():
                    sample_idx = int(user_input)
                    if 0 <= sample_idx < len(tester.test_dataset):
                        tester.test_specific_samples([sample_idx])
                    else:
                        print(f"❌ 索引应在 0-{len(tester.test_dataset)-1} 范围内")
                else:
                    print("❌ 请输入有效数字")
                    
            except KeyboardInterrupt:
                print("\n👋 测试结束")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                
    except Exception as e:
        logger.error(f"测试失败: {e}")

if __name__ == "__main__":
    main()