//
//  CameraView.swift
//  CornMildewDetector
//
//  Created by 何哈哈 on 2026/4/22.
//

import SwiftUI
import AVFoundation
import Vision

// MARK: - 主相机视图
struct CameraView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var detector = DetectorService()
    
    var body: some View {
        ZStack {
            CameraPreviewView(detector: detector)
                .ignoresSafeArea()  // ✅ 修复：edgesIgnoringSafeArea 已废弃
            
            DetectionOverlayView(detections: detector.detections)
                .ignoresSafeArea()  // ✅ 修复：edgesIgnoringSafeArea 已废弃
            
            // 顶部状态栏
            VStack {
                HStack {
                    Button(action: { dismiss() }) {
                        Image(systemName: "xmark")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(.white)
                            .padding(12)
                            .background(Color.black.opacity(0.6))
                            .clipShape(Circle())
                    }
                    .padding(.leading, 16)
                    
                    Spacer()
                    
                    HStack(spacing: 6) {
                        Circle()
                            .fill(detector.isModelLoaded ? Color.green : Color.red)
                            .frame(width: 8, height: 8)
                        Text(detector.isModelLoaded ? "CoreML 就绪" : "加载中...")
                            .font(.caption)
                            .foregroundColor(.white)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.black.opacity(0.6))
                    .cornerRadius(20)
                    .padding(.trailing, 16)
                }
                .padding(.top, 50)
                
                Spacer()
            }
            
            // 底部统计面板
            VStack {
                Spacer()
                HStack {
                    Image(systemName: "chart.bar.fill")
                        .foregroundColor(.white.opacity(0.8))
                    Text(detector.statsText)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                    Spacer()
                }
                .padding()
                .background(Color.black.opacity(0.7))
                .cornerRadius(12)
                .padding(.horizontal, 16)
                .padding(.bottom, 40)
            }
        }
    }
}

// MARK: - 相机预览 (UIViewRepresentable)
struct CameraPreviewView: UIViewRepresentable {
    @ObservedObject var detector: DetectorService
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        view.backgroundColor = .black
        
        let cameraVC = CameraViewController()
        cameraVC.detector = detector
        cameraVC.view.frame = view.bounds
        cameraVC.view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        
        view.addSubview(cameraVC.view)
        context.coordinator.cameraVC = cameraVC
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {}
    
    func makeCoordinator() -> Coordinator { Coordinator() }
    
    class Coordinator {
        var cameraVC: CameraViewController?
    }
}

// MARK: - 相机控制器 (AVCapture)
class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var detector: DetectorService?
    
    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private let videoOutput = AVCaptureVideoDataOutput()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
    }
    
    private func setupCamera() {
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let videoInput = try? AVCaptureDeviceInput(device: videoDevice),
              captureSession.canAddInput(videoInput) else {
            print("无法访问后置摄像头")
            return
        }
        
        captureSession.beginConfiguration()
        captureSession.addInput(videoInput)
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer?.videoGravity = .resizeAspectFill
        previewLayer?.frame = view.bounds
        view.layer.addSublayer(previewLayer!)
        
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        captureSession.commitConfiguration()
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        detector?.detect(pixelBuffer: pixelBuffer)
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }
}

// MARK: - 检测框叠加层 (半透明版本)
struct DetectionOverlayView: View {
    let detections: [DetectionResult]
    
    var body: some View {
        GeometryReader { geometry in
            ForEach(detections) { detection in
                let original = detection.box
                
                // 微调偏移量 (根据实际效果调整)
                let offsetX: CGFloat = 0.085
                let offsetY: CGFloat = 0.0
                
                let rotatedBox = CGRect(
                    x: original.origin.y + offsetX,
                    y: 1 - original.origin.x - original.width + offsetY,
                    width: original.height,
                    height: original.width
                )
                
                let rect = VNImageRectForNormalizedRect(
                    rotatedBox,
                    Int(geometry.size.width),
                    Int(geometry.size.height)
                )
                
                let color = Color(detection.displayColor)
                
                ZStack(alignment: .topLeading) {
                    // 框：半透明填充 + 边框
                    Rectangle()
                        .fill(color.opacity(0.2))
                        .stroke(color.opacity(0.8), lineWidth: 2)
                        .frame(width: rect.width, height: rect.height)
                    
                    // 标签：文字纯白不透明，背景半透明
                    Text("\(detection.className) \(Int(detection.confidence * 100))%")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(color.opacity(0.55))
                        .cornerRadius(4)
                        .offset(y: -24)
                }
                .position(x: rect.midX, y: rect.midY)
            }
        }
    }
}

// MARK: - 坐标转换工具 (Vision → UIKit)
func VNImageRectForNormalizedRect(_ normalizedRect: CGRect, _ imageWidth: Int, _ imageHeight: Int) -> CGRect {
    let x = normalizedRect.origin.x * CGFloat(imageWidth)
    let y = (1 - normalizedRect.origin.y - normalizedRect.height) * CGFloat(imageHeight)
    let width = normalizedRect.width * CGFloat(imageWidth)
    let height = normalizedRect.height * CGFloat(imageHeight)
    return CGRect(x: x, y: y, width: width, height: height)
}
