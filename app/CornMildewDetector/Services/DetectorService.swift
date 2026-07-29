//
//  DetectorService.swift
//  CornMildewDetector
//
//  二分类版本：light_mold / heavy_mold
//  修复：霉变率计算公式（轻度×0.5 + 重度×1）/ 总粒数
//

import Foundation
import Vision
import CoreML
import UIKit

// MARK: - 检测结果
public struct DetectionResult: Identifiable {
    public let id = UUID()
    public let box: CGRect
    public let confidence: Float
    public let classId: Int
    public let className: String
    
    public var displayColor: UIColor {
        switch classId {
        case 0: return .systemOrange
        case 1: return .systemRed
        default: return .gray
        }
    }
    
    public init(box: CGRect, confidence: Float, classId: Int, className: String) {
        self.box = box
        self.confidence = confidence
        self.classId = classId
        self.className = className
    }
}

// MARK: - 检测服务
public class DetectorService: ObservableObject {
    
    // 可调参数
    private let confidenceThreshold: Float = 0.35
    private let nmsIouThreshold: Float = 0.45
    
    @Published public var detections: [DetectionResult] = []
    @Published public var statsText: String = "等待检测..."
    @Published public var isModelLoaded: Bool = false
    @Published public var errorMessage: String?
    
    private var visionModel: VNCoreMLModel?
    
    // MARK: - 霉变率计算参数（根据预统计结果配置）
    /// 100g 玉米粒的平均粒数（需要通过预统计实验得出）
    private let averageGrainsPer100g: Float = 110
    /// 一斤 = 500g = 5 × 100g，所以一斤 ≈ 5μ 粒
    private var totalGrainsPerJin: Float {
        return averageGrainsPer100g * 5
    }
    
    public init() {
        loadModel()
    }
    
    // MARK: - 模型加载
    private func loadModel() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            self.printBundleContents()
            
            let candidates: [(name: String, ext: String?)] = [
                ("best", nil),
                ("best", "mlpackage"),
                ("best", "mlmodelc"),
                ("best", "mlmodel")
            ]
            
            var loaded = false
            
            for candidate in candidates {
                guard let modelURL = Bundle.main.url(forResource: candidate.name, withExtension: candidate.ext) else {
                    continue
                }
                
                do {
                    print("🔄 尝试加载: \(modelURL.lastPathComponent)")
                    let coreMLModel = try MLModel(contentsOf: modelURL)
                    self.visionModel = try VNCoreMLModel(for: coreMLModel)
                    loaded = true
                    
                    DispatchQueue.main.async {
                        self.isModelLoaded = true
                        self.errorMessage = nil
                        self.statsText = "模型就绪 ✅"
                        print("✅ 模型加载成功: \(modelURL.lastPathComponent)")
                    }
                    break
                    
                } catch {
                    print("⚠️ 加载 \(modelURL.lastPathComponent) 失败: \(error.localizedDescription)")
                }
            }
            
            if !loaded {
                DispatchQueue.main.async {
                    self.isModelLoaded = false
                    self.errorMessage = "找不到模型文件，请检查 best.mlpackage 是否已添加到项目"
                    self.statsText = "❌ 模型未找到"
                    print("❌ 所有模型加载尝试均失败")
                }
            }
        }
    }
    
    // MARK: - 打印 Bundle 内容
    private func printBundleContents() {
        let bundlePath = Bundle.main.bundlePath
        print("📁 Bundle 路径: \(bundlePath)")
        
        do {
            let contents = try FileManager.default.contentsOfDirectory(atPath: bundlePath)
            let modelFiles = contents.filter {
                $0.contains("best") || $0.contains("mlmodel") || $0.contains("mlpackage")
            }
            if modelFiles.isEmpty {
                print("⚠️ Bundle 中没有找到任何模型文件")
            } else {
                print("📁 找到相关文件: \(modelFiles.joined(separator: ", "))")
            }
        } catch {
            print("⚠️ 无法读取 Bundle 内容: \(error)")
        }
    }
    
    // MARK: - 核心检测方法
    public func detect(pixelBuffer: CVPixelBuffer) {
        guard let visionModel = visionModel else {
            DispatchQueue.main.async {
                self.statsText = "⚠️ 模型未加载"
            }
            return
        }
        
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, error in
            if let error = error {
                print("检测错误: \(error)")
                DispatchQueue.main.async {
                    self?.statsText = "检测失败: \(error.localizedDescription)"
                }
                return
            }
            self?.handleResults(request.results)
        }
        
        request.imageCropAndScaleOption = .centerCrop
        
        do {
            try VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
        } catch {
            print("执行检测失败: \(error)")
            DispatchQueue.main.async {
                self.statsText = "执行检测失败"
            }
        }
    }
    
    // MARK: - 结果处理
    private func handleResults(_ results: [Any]?) {
        guard let observations = results as? [VNRecognizedObjectObservation] else {
            DispatchQueue.main.async {
                self.detections = []
                self.statsText = "未检测到目标"
            }
            return
        }
        
        var rawDetections: [DetectionResult] = []
        
        for obs in observations {
            guard let label = obs.labels.first else { continue }
            guard label.confidence >= confidenceThreshold else { continue }
            
            let identifier = label.identifier.lowercased()
            let classId: Int
            let className: String
            
            if identifier.contains("heavy") || identifier.contains("重度") || identifier.hasPrefix("1") {
                classId = 1
                className = "重度霉变"
            } else if identifier.contains("light") || identifier.contains("轻度") || identifier.hasPrefix("0") {
                classId = 0
                className = "轻度霉变"
            } else {
                classId = 0
                className = "轻度霉变"
            }
            
            let detection = DetectionResult(
                box: obs.boundingBox,
                confidence: label.confidence,
                classId: classId,
                className: className
            )
            rawDetections.append(detection)
        }
        
        let finalDetections = applyNMS(on: rawDetections, iouThreshold: nmsIouThreshold)
        
        // 统计数量
        let finalLight = finalDetections.filter { $0.classId == 0 }.count
        let finalHeavy = finalDetections.filter { $0.classId == 1 }.count
        
        // ✅ 核心修复：计算霉变率
        // 轻度霉变只损失一半重量 → 计 0.5 粒
        // 重度霉变整粒丢弃 → 计 1 粒
        let mildWeight: Float = Float(finalLight) * 0.5
        let heavyWeight: Float = Float(finalHeavy) * 1.0
        let totalLossWeight = mildWeight + heavyWeight
        
        // 总粒数（基于预统计，一斤 ≈ 5μ 粒）
        let totalGrains = totalGrainsPerJin
        
        // 霉变率（百分比）
        let mildewRate = totalGrains > 0 ? (totalLossWeight / totalGrains) * 100 : 0
        
        // 判定是否超标（霉变率 > 5% 视为超标）
        let isExceeded = mildewRate > 5.0
        
        DispatchQueue.main.async {
            self.detections = finalDetections
            
            let statusIcon = isExceeded ? "🚨" : "✅"
            let statusText = isExceeded ? "超标" : "合格"
            
            self.statsText = """
            轻度: \(finalLight) | 重度: \(finalHeavy) | 总数: \(finalLight + finalHeavy)
            霉变率: \(String(format: "%.1f", mildewRate))% \(statusIcon) \(statusText)
            折算损失: 轻度 \(String(format: "%.1f", mildWeight)) 粒 + 重度 \(String(format: "%.1f", heavyWeight)) 粒 = \(String(format: "%.1f", totalLossWeight)) 粒
            """
        }
    }
    
    // MARK: - 解析类别ID
    private func parseClassId(from identifier: String) -> Int {
        let lower = identifier.lowercased()
        if lower.contains("heavy") || lower.contains("重度") || lower.hasPrefix("1") {
            return 1
        } else if lower.contains("light") || lower.contains("轻度") || lower.hasPrefix("0") {
            return 0
        }
        return 0
    }
    
    // MARK: - NMS
    private func applyNMS(on detections: [DetectionResult], iouThreshold: Float) -> [DetectionResult] {
        guard detections.count > 1 else { return detections }
        
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var keep: [DetectionResult] = []
        var indices = Array(0..<sorted.count)
        
        while !indices.isEmpty {
            let first = indices.removeFirst()
            keep.append(sorted[first])
            
            let firstBox = sorted[first].box
            let remaining = indices
            indices.removeAll()
            
            for idx in remaining {
                let iou = computeIOU(firstBox, sorted[idx].box)
                if iou < iouThreshold {
                    indices.append(idx)
                }
            }
        }
        
        return keep
    }
    
    private func computeIOU(_ box1: CGRect, _ box2: CGRect) -> Float {
        let intersection = box1.intersection(box2)
        if intersection.isNull { return 0 }
        
        let interArea = intersection.width * intersection.height
        let area1 = box1.width * box1.height
        let area2 = box2.width * box2.height
        let unionArea = area1 + area2 - interArea
        
        guard unionArea > 0 else { return 0 }
        return Float(interArea / unionArea)
    }
    
    // MARK: - 重置
    public func reset() {
        DispatchQueue.main.async {
            self.detections = []
            self.statsText = "已重置"
        }
    }
}
// ✅ 文件结束，没有 View 相关代码
