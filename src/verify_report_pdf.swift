// Inspect the actual final PDF with system PDFKit; no third-party packages.
import Foundation
import PDFKit
import CoreGraphics
import AppKit

final class ImageCollector {
    let page: Int
    var images: [[String: Int]] = []
    init(page: Int) { self.page = page }
}

func inspectResources(_ dictionary: CGPDFDictionaryRef, _ info: UnsafeMutableRawPointer) {
    var resources: CGPDFDictionaryRef?
    var objects: CGPDFDictionaryRef?
    if CGPDFDictionaryGetDictionary(dictionary, "Resources", &resources),
       let resources = resources,
       CGPDFDictionaryGetDictionary(resources, "XObject", &objects), let objects = objects {
        CGPDFDictionaryApplyFunction(objects, inspectObject, info)
    }
}

func inspectObject(_ key: UnsafePointer<CChar>, _ object: CGPDFObjectRef, _ info: UnsafeMutableRawPointer?) {
    guard let info = info else { return }
    var stream: CGPDFStreamRef?
    guard CGPDFObjectGetValue(object, .stream, &stream), let actualStream = stream,
          let dictionary = CGPDFStreamGetDictionary(actualStream) else { return }
    var subtype: UnsafePointer<CChar>?
    guard CGPDFDictionaryGetName(dictionary, "Subtype", &subtype), let subtypeName = subtype else { return }
    let name = String(cString: subtypeName)
    if name == "Image" {
        var width: CGPDFInteger = 0
        var height: CGPDFInteger = 0
        if CGPDFDictionaryGetInteger(dictionary, "Width", &width),
           CGPDFDictionaryGetInteger(dictionary, "Height", &height), width > 100, height > 100 {
            let collector = Unmanaged<ImageCollector>.fromOpaque(info).takeUnretainedValue()
            collector.images.append(["page": collector.page, "width_pixels": width, "height_pixels": height])
        }
    } else if name == "Form" {
        inspectResources(dictionary, info)
    }
}

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: "FinalPDFVerification", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
}

do {
    let rawArguments = CommandLine.arguments
    // Older system Swift exposes interpreter flags before the script's -- separator.
    let args: [String]
    if let separator = rawArguments.firstIndex(of: "--") {
        args = [rawArguments[0]] + Array(rawArguments.dropFirst(separator + 1))
    } else {
        args = rawArguments
    }
    try require(args.count == 4, "Usage: swift src/verify_report_pdf.swift PDF_PATH PROJECT_ROOT OUTPUT_JSON; received \(args)")
    let pdfURL = URL(fileURLWithPath: args[1])
    let root = URL(fileURLWithPath: args[2], isDirectory: true)
    guard let document = PDFDocument(url: pdfURL) else {
        throw NSError(domain: "FinalPDFVerification", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cannot open PDF"])
    }
    let text = document.string ?? ""
    let extractedTextPath = args[3] + ".txt"
    if !FileManager.default.fileExists(atPath: extractedTextPath) {
        try text.write(toFile: extractedTextPath, atomically: false, encoding: .utf8)
    }
    let normalized = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }.joined(separator: " ")
        .replacingOccurrences(of: "\\s*_\\s*", with: "_", options: .regularExpression)
    let requiredText = [
        "https://github.com/tagbondy/sweettv-recommender",
        "outputs/submission_v5.csv",
        "18b79a44e38d3a19b0edd75587535161c32b29bcadd19d28ad106a4c7cc651c0",
        "13571199046623737884", "4478271560731083975",
        "Figure 1. Best SWEET.TV Kaggle submission (V5): Public MAP@5 = 0.44671; Private MAP@5 = 0.43927.",
        "Figure 2. Complete Kaggle submission history, V1–V5, demonstrating the progression from Association Rules through personalized ranking and the final V5 ensemble.",
        "0.31901", "0.31614", "0.41082", "0.39407", "0.43568", "0.43224", "0.44087", "0.43552", "0.44671", "0.43927",
        "0.50 was not achieved"
    ]
    for token in requiredText { try require(normalized.contains(token), "Required PDF text is missing: \(token)") }
    for obsolete in ["V5 has no measured Kaggle score", "V5 is awaiting manual submission", "V4 is the scored incumbent", "DRAFT —"] {
        try require(!normalized.contains(obsolete), "Obsolete draft claim remains in final PDF: \(obsolete)")
    }
    var images: [[String: Int]] = []
    var pageSizes: [[String: Any]] = []
    for index in 0..<document.pageCount {
        guard let page = document.page(at: index), let reference = page.pageRef,
              let dictionary = reference.dictionary else { continue }
        let collector = ImageCollector(page: index + 1)
        inspectResources(dictionary, Unmanaged.passUnretained(collector).toOpaque())
        images += collector.images
        let bounds = page.bounds(for: .mediaBox)
        pageSizes.append(["page": index + 1, "width_points": bounds.width, "height_points": bounds.height])
    }
    var evidence: [[String: Any]] = []
    for filename in ["kaggle_v5_scores.png", "kaggle_all_scores.png"] {
        let imageURL = root.appendingPathComponent("reports/evidence/" + filename)
        let data = try Data(contentsOf: imageURL)
        guard let bitmap = NSBitmapImageRep(data: data) else {
            throw NSError(domain: "FinalPDFVerification", code: 3, userInfo: [NSLocalizedDescriptionKey: "Cannot decode original PNG \(filename)"])
        }
        let matches = images.filter { $0["width_pixels"] == bitmap.pixelsWide && $0["height_pixels"] == bitmap.pixelsHigh }
        try require(!matches.isEmpty, "No actual PDF image matches original dimensions for \(filename)")
        evidence.append(["filename": filename, "original_width_pixels": bitmap.pixelsWide,
                         "original_height_pixels": bitmap.pixelsHigh, "matching_pdf_images": matches])
    }
    let result: [String: Any] = ["pdf": pdfURL.path, "page_count": document.pageCount,
                                "required_text_verified": requiredText, "obsolete_claims_absent": true,
                                "embedded_images": images, "screenshot_image_resources_verified": evidence,
                                "pages": pageSizes, "visual_inspection_required_separately": true,
                                "verification_method": "System PDFKit text and CGPDF image resource inspection; original PNG dimensions via AppKit"]
    let output = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
    try output.write(to: URL(fileURLWithPath: args[3]), options: .withoutOverwriting)
    print("Actual PDF verified: \(document.pageCount) pages; both screenshot image resources and captions found.")
} catch {
    FileHandle.standardError.write(Data(("PDF verification failed: \(error.localizedDescription)\n").utf8))
    exit(1)
}
