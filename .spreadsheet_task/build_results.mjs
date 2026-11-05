import fs from 'node:fs/promises';
import path from 'node:path';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const projectDir = path.resolve(import.meta.dirname, '..');
const workDir = path.join(projectDir, 'reports', '.simuletic_dms_work');
const report = JSON.parse(await fs.readFile(path.join(workDir, 'predictions.json'), 'utf8'));
const outputDir = path.join(projectDir, 'reports');
const outputPath = path.join(outputDir, 'simuletic_dms_yawn_results.xlsx');
const workbook = Workbook.create();

const valid = report.rows.filter((r) => r.p_yawn !== null);
const yawn = valid.filter((r) => r.prediction === '하품').length;
const noYawn = valid.filter((r) => r.prediction === '하품 아님').length;
const unknown = report.rows.length - valid.length;
const statusCounts = new Map();
for (const row of report.rows) statusCounts.set(row.status, (statusCounts.get(row.status) ?? 0) + 1);

const summary = workbook.worksheets.add('요약');
summary.showGridlines = false;
summary.getRange('A1:F1').merge();
summary.getRange('A1').values = [['Simuletic DMS 하품 모델 추론 결과']];
summary.getRange('A1:F1').format = {
  fill: '#12304A', font: { bold: true, color: '#FFFFFF', size: 18 },
  verticalAlignment: 'center', horizontalAlignment: 'left',
};
summary.getRange('A1:F1').format.rowHeight = 38;
summary.getRange('A3:B7').values = [
  ['항목', '결과'],
  ['입력 사진 수', report.count],
  ['분류 완료', valid.length],
  ['하품 예측', yawn],
  ['하품 아님 예측', noYawn],
];
summary.getRange('A8:B8').values = [['판정 불가', unknown]];
summary.getRange('A3:B3').format = { fill: '#DCE9F2', font: { bold: true, color: '#12304A' } };
summary.getRange('A4:B8').format = { borders: { preset: 'inside', style: 'thin', color: '#DCE3EA' } };
summary.getRange('A10:B13').values = [
  ['설정/상태', '값'],
  ['모델', path.relative(projectDir, report.weights)],
  ['판정 기준 P(yawn)', report.threshold],
  ['검출 상태: no_face', statusCounts.get('no_face') ?? 0],
];
summary.getRange('A10:B10').format = { fill: '#DCE9F2', font: { bold: true, color: '#12304A' } };
summary.getRange('A10:B13').format.borders = { preset: 'inside', style: 'thin', color: '#DCE3EA' };
summary.getRange('A15:F17').merge();
summary.getRange('A15').values = [[
  '참고: 각 사진을 독립적으로 분류한 모델 예측입니다. 원본 파일명은 정답 라벨로 간주하지 않았으므로 정확도나 혼동행렬은 산출하지 않았습니다. 얼굴을 정확히 1명 검출하지 못한 사진은 판정 불가로 남겼습니다.',
]];
summary.getRange('A15:F17').format = {
  fill: '#FFF4D6', font: { color: '#6B5317', size: 11 }, wrapText: true,
  verticalAlignment: 'center',
};
summary.getRange('A:A').format.columnWidth = 29;
summary.getRange('B:B').format.columnWidth = 66;
summary.getRange('C:F').format.columnWidth = 15;
summary.getRange('A15:F17').format.rowHeight = 26;
summary.freezePanes.freezeRows(3);

const detail = workbook.worksheets.add('사진별 결과');
detail.showGridlines = false;
detail.getRange('A1:G1').merge();
detail.getRange('A1').values = [['사진별 모델 예측']];
detail.getRange('A1:G1').format = {
  fill: '#12304A', font: { bold: true, color: '#FFFFFF', size: 16 }, verticalAlignment: 'center',
};
detail.getRange('A1:G1').format.rowHeight = 34;
detail.getRange('A2:G2').merge();
detail.getRange('A2').values = [[`확률 기준: ${report.threshold.toFixed(2)} | 사진 ${report.count}장 | 판정은 모델 출력이며 정답 라벨이 아닙니다.`]];
detail.getRange('A2:G2').format = { fill: '#EAF1F6', font: { color: '#334155', size: 10 } };
detail.getRange('A4:G4').values = [[
  '원본 사진', '모델 입력 입 영역', '파일명', '예측', '하품 확률', '검출 상태', '원본 경로',
]];
detail.getRange(`A4:G${report.rows.length + 4}`).format.font = { name: 'Arial', size: 10 };
detail.getRange('A4:G4').format = {
  fill: '#226082', font: { bold: true, color: '#FFFFFF', size: 10 },
  wrapText: true, horizontalAlignment: 'center', verticalAlignment: 'center',
};
detail.getRange('A4:G4').format.rowHeight = 28;
const data = report.rows.map((r) => [null, null, r.file, r.prediction, r.p_yawn, r.status, r.source]);
if (data.length) detail.getRange(`A5:G${data.length + 4}`).values = data;
detail.getRange(`A5:G${data.length + 4}`).format = {
  verticalAlignment: 'center', wrapText: false,
  borders: { preset: 'inside', style: 'thin', color: '#E2E8F0' },
};
detail.getRange(`D5:D${data.length + 4}`).format.horizontalAlignment = 'center';
detail.getRange(`E5:E${data.length + 4}`).format.numberFormat = '0.0%';
detail.getRange(`E5:E${data.length + 4}`).format.horizontalAlignment = 'center';
detail.getRange(`F5:F${data.length + 4}`).format.horizontalAlignment = 'center';
detail.getRange(`A:A`).format.columnWidthPx = 118;
detail.getRange(`B:B`).format.columnWidthPx = 94;
detail.getRange(`C:C`).format.columnWidth = 32;
detail.getRange(`D:D`).format.columnWidth = 18;
detail.getRange(`E:E`).format.columnWidth = 16;
detail.getRange(`F:F`).format.columnWidth = 22;
detail.getRange(`G:G`).format.columnWidth = 72;
detail.getRange(`A5:G${data.length + 4}`).format.rowHeightPx = 82;
detail.freezePanes.freezeRows(4);
detail.tables.add(`A4:G${data.length + 4}`, true, 'SimuleticPredictions');

for (let i = 0; i < report.rows.length; i++) {
  const row = report.rows[i];
  const anchorRow = i + 4; // zero-based worksheet row for Excel row 5 + i
  const original = await fs.readFile(row.original_thumb);
  detail.images.add({
    dataUrl: `data:image/jpeg;base64,${original.toString('base64')}`,
    anchor: { from: { row: anchorRow, col: 0 }, extent: { widthPx: 112, heightPx: 75 } },
  });
  if (row.mouth_thumb) {
    const crop = await fs.readFile(row.mouth_thumb);
    detail.images.add({
      dataUrl: `data:image/jpeg;base64,${crop.toString('base64')}`,
      anchor: { from: { row: anchorRow, col: 1 }, extent: { widthPx: 72, heightPx: 72 } },
    });
  }
}

workbook.recalculate();
const summaryPreview = await workbook.render({ sheetName: '요약', autoCrop: 'all', scale: 1, format: 'png' });
await fs.writeFile(path.join(workDir, 'summary_preview.png'), new Uint8Array(await summaryPreview.arrayBuffer()));
const detailPreview = await workbook.render({ sheetName: '사진별 결과', range: 'A1:G9', scale: 1, format: 'png' });
await fs.writeFile(path.join(workDir, 'detail_preview.png'), new Uint8Array(await detailPreview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(JSON.stringify({ outputPath, rows: report.rows.length, bytes: (await fs.stat(outputPath)).size }));
