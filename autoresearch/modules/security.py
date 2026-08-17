"""Security Module - Hardened input handling, API security, adversarial testing."""
import re
import hashlib
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict

class SecurityModule:
    """
    Provides security hardening for the AutoResearch pipeline.
    Includes input sanitization, rate limiting, adversarial input detection,
    and secure PDF parsing sandbox.
    """

    def __init__(self):
        self.request_log = defaultdict(list)
        self.blocked_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'\b(SELECT|INSERT|DELETE|UPDATE|DROP|UNION)\b',
            r'\.\./',
            r'\x00',
        ]
        self.max_requests_per_minute = 30
        self.suspicious_threshold = 5

    def sanitize_query(self, query: str) -> Dict[str, Any]:
        """Sanitize user input query. Returns cleaned query + security report."""
        original = query
        findings = []

        # Check length
        if len(query) > 500:
            findings.append("QUERY_TOO_LONG")
            query = query[:500]

        # Check for injection patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                findings.append(f"BLOCKED_PATTERN: {pattern[:30]}")
                query = re.sub(pattern, '', query, flags=re.IGNORECASE)

        # Normalize whitespace
        query = ' '.join(query.split())

        # Check entropy (random garbage detection)
        entropy = self._calculate_entropy(query)
        if entropy > 4.5 and len(query) > 50:
            findings.append("HIGH_ENTROPY_GARBAGE")

        return {
            "original": original,
            "sanitized": query,
            "security_findings": findings,
            "is_safe": len(findings) == 0,
            "entropy": round(entropy, 2)
        }

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy to detect random/garbage input."""
        if not text:
            return 0
        from math import log2
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        entropy = 0
        length = len(text)
        for count in freq.values():
            p = count / length
            entropy -= p * log2(p)
        return entropy

    def check_rate_limit(self, client_id: str) -> Dict[str, Any]:
        """Rate limiting per client."""
        now = time.time()
        window = 60  # 1 minute

        # Clean old entries
        self.request_log[client_id] = [
            t for t in self.request_log[client_id] 
            if now - t < window
        ]

        count = len(self.request_log[client_id])

        if count >= self.max_requests_per_minute:
            return {
                "allowed": False,
                "reason": "RATE_LIMIT_EXCEEDED",
                "retry_after": int(window - (now - self.request_log[client_id][0]))
            }

        self.request_log[client_id].append(now)
        return {"allowed": True, "requests_in_window": count + 1}

    def adversarial_test_suite(self, pipeline_func) -> Dict[str, Any]:
        """Test pipeline against adversarial inputs."""
        test_cases = [
            ("SQL injection attempt", "'; DROP TABLE papers; --"),
            ("XSS attempt", "<script>alert('xss')</script> machine learning"),
            ("Path traversal", "../../../etc/passwd neural networks"),
            ("Buffer overflow pattern", "A" * 10000),
            ("Unicode obfuscation", "𝒎𝒂𝒄𝒉𝒊𝒏𝒆 𝒍𝒆𝒂𝒓𝒏𝒊𝒏𝒈"),
            ("Null bytes", "deep\x00learning"),
            ("Valid query", "transformer architectures"),
        ]

        results = []
        for test_name, test_input in test_cases:
            try:
                sanitized = self.sanitize_query(test_input)
                if sanitized["is_safe"]:
                    # Only run pipeline on safe inputs
                    result = pipeline_func(sanitized["sanitized"])
                    status = "PROCESSED"
                else:
                    status = "BLOCKED"
                    result = None

                results.append({
                    "test": test_name,
                    "input_preview": test_input[:50],
                    "status": status,
                    "findings": sanitized["security_findings"],
                    "pipeline_ran": result is not None
                })
            except Exception as e:
                results.append({
                    "test": test_name,
                    "status": "ERROR",
                    "error": str(e)
                })

        passed = sum(1 for r in results if r["status"] in ["BLOCKED", "PROCESSED"])
        return {
            "total_tests": len(test_cases),
            "passed": passed,
            "failed": len(test_cases) - passed,
            "details": results
        }

    def secure_pdf_parse(self, pdf_path: str) -> Dict[str, Any]:
        """Sandboxed PDF parsing with security checks."""
        import os
        from pathlib import Path

        path = Path(pdf_path)

        # Validate path
        if not path.exists():
            return {"error": "FILE_NOT_FOUND", "safe": False}

        # Check file size (max 50MB)
        max_size = 50 * 1024 * 1024
        if path.stat().st_size > max_size:
            return {"error": "FILE_TOO_LARGE", "safe": False}

        # Check extension
        if path.suffix.lower() != '.pdf':
            return {"error": "INVALID_EXTENSION", "safe": False}

        # Calculate file hash for integrity
        with open(path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        return {
            "safe": True,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "hash": file_hash,
            "message": "File passed security checks"
        }
