import Foundation
import Vision
import CoreML
import UIKit

struct DetectionResult: Identifiable {
    let id = UUID()
    let box: CGRect
    let confidence: Float
    let classId: Int
    let className: String
    
    var displayColor: UIColor {
        switch classId {
        case 0: return .systemOrange   // light_mold
        case 1: return .systemRed      // heavy_mold
        default: return .gray
        }
    }
}

class DetectorService: ObservableObject {
    @Published var detections: [DetectionResult] = []
    @Published var statsText: String = "等待检测..."
    @Published var isModelLoaded: Bool = false
    
    private var visionModel: VNCoreMLModel?
    private let classNames = ["轻度霉变", "重度霉变"]
    private var lightCount = 0, heavyCount = 0
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            do {
                guard let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlpackage") else {
                    print("找不到 best.mlpackage")
                    return
                }
                let coreMLModel = try MLModel(contentsOf: modelURL)
                self.visionModel = try VNCoreMLModel(for: coreMLModel)
                DispatchQueue.main.async { self.isModelLoaded = true }
            } catch {
                print("模型加载失败: \(error)")
            }
        }
    }
    
    func detect(pixelBuffer: CVPixelBuffer) {
        guard let visionModel = visionModel else { return }
        let request = VNCoreMLRequest(model: visionModel) { [weak self] request, _ in
            self?.handleResults(request.results)
        }
        request.imageCropAndScaleOption = .centerCrop
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer).perform([request])
    }
    
    private func handleResults(_ results: [Any]?) {
        guard let observations = results as? [VNRecognizedObjectObservation] else { return }
        var detections: [DetectionResult] = []
        var light = 0, heavy = 0
        
        for obs in observations {
            guard let label = obs.labels.first else { continue }
            let classId = label.identifier == "heavy_mold" ? 1 : 0
            if classId == 0 { light += 1 } else { heavy += 1 }
            detections.append(DetectionResult(
                box: obs.boundingBox,
                confidence: label.confidence,
                classId: classId,
                className: classNames[classId]
            ))
        }
        
        DispatchQueue.main.async { [weak self] in
            self?.detections = detections
            self?.lightCount = light
            self?.heavyCount = heavy
            let total = light + heavy
            let rate = total > 0 ? Float(heavy) / Float(total) * 100 : 0
            self?.statsText = "轻度: \(light) | 重度: \(heavy)\n霉变率: \(String(format: "%.1f", rate))%"
        }
    }
}
