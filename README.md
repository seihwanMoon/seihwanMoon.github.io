# Moon's Blog

엔지니어의 개인 기록 블로그입니다.

## 운영 방식

이 저장소는 `obsidian-webpage-export`만으로 운영합니다.

- 작성 Vault: `G:\My Drive\Bobsidian\github.vault`
- 배포 저장소: `G:\My Drive\Bobsidian\seihwanMoon.github.io`
- 공개 주소: `https://seihwanmoon.github.io/`

## 배포 절차

1. Obsidian에서 `github.vault`를 엽니다.
2. 처음 한 번은 `Set html export settings`를 열고, file picker에서 공개할 파일과 폴더가 선택되어 있는지 확인한 뒤 `Save`를 누릅니다.
3. 그 다음 `Export using previous settings`를 실행합니다.
4. 생성된 HTML이 이 저장소에 직접 반영되면 검토 후 GitHub에 push 합니다.

추가 Node 패키지나 별도 후처리 프로그램은 현재 필요하지 않습니다.

## Export 후 체크리스트

### 1. export 직후 확인

- `G:\My Drive\Bobsidian\seihwanMoon.github.io` 안에 방금 수정한 문서의 HTML이 생성되었는지 확인합니다.
- `index.html`, `topics.html`, 주요 주제 페이지가 갱신되었는지 확인합니다.
- 이미지, 첨부파일, 다이어그램이 필요한 문서라면 해당 페이지를 직접 열어 깨지지 않았는지 봅니다.
- `site-lib\\metadata.json` 안에 `webpages`가 비어 있지 않은지 확인합니다. 비어 있으면 file picker 저장이 안 된 상태입니다.

### 2. push 전 확인

- `git status`로 의도한 파일만 변경되었는지 확인합니다.
- 원하지 않는 임시 파일이 포함되지 않았는지 확인합니다.
- 변경 내용이 맞으면 commit 후 push 합니다.

예시:

```powershell
cd "G:\My Drive\Bobsidian\seihwanMoon.github.io"
git status
git add .
git commit -m "docs: update published notes"
git push
```

### 3. 공개 사이트 확인

- GitHub Pages 반영 후 `https://seihwanmoon.github.io/`에 접속합니다.
- 홈, 주제 탐색, 최근 갱신 페이지가 정상적으로 열리는지 확인합니다.
- 내부 링크, 검색, 태그, 파일 트리가 의도대로 동작하는지 확인합니다.

Powered by [KosmosisDire/obsidian-webpage-export](https://github.com/KosmosisDire/obsidian-webpage-export)  
Hosted by GitHub Pages
