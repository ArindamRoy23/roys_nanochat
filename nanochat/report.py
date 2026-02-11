"""
Report generation module for nanochat training runs.
Stores markdown sections and generates a final report.
"""

import os
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from nanochat.common import get_base_dir


class Report:
    """Manages markdown report sections for a training run."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or get_base_dir()
        self.report_dir = os.path.join(self.base_dir, "report")
        os.makedirs(self.report_dir, exist_ok=True)
        self.sections: Dict[str, List[str]] = {}
    
    def log(self, section: str, data: List[Any]):
        """Log data to a report section.
        
        Args:
            section: Section name (e.g., "Tokenizer training")
            data: List of data items (dicts, strings, etc.)
        """
        if section not in self.sections:
            self.sections[section] = []
        
        # Convert data items to markdown strings
        for item in data:
            if isinstance(item, str):
                self.sections[section].append(item)
            elif isinstance(item, dict):
                # Convert dict to markdown
                lines = []
                for key, value in item.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"- **{key}**: {value}")
                    elif isinstance(value, str):
                        lines.append(f"- **{key}**: {value}")
                    else:
                        lines.append(f"- **{key}**: {json.dumps(value)}")
                self.sections[section].extend(lines)
            else:
                self.sections[section].append(str(item))
        
        # Write section to file
        section_file = os.path.join(self.report_dir, f"{section.lower().replace(' ', '_')}.md")
        with open(section_file, "w") as f:
            f.write(f"# {section}\n\n")
            f.write("\n".join(self.sections[section]))
            f.write("\n")
    
    def reset(self):
        """Reset the report directory and write header with system info."""
        # Clear existing sections
        self.sections = {}
        
        # Remove existing report files
        for file in Path(self.report_dir).glob("*.md"):
            file.unlink()
        
        # Write header with system info
        header_lines = [
            "# Nanochat Training Run Report",
            "",
            f"**Started**: {datetime.now().isoformat()}",
            "",
            "## System Information",
            "",
        ]
        
        # Add system info
        header_lines.append(f"- **Platform**: {platform.platform()}")
        header_lines.append(f"- **Python**: {platform.python_version()}")
        header_lines.append(f"- **Machine**: {platform.machine()}")
        header_lines.append(f"- **Processor**: {platform.processor()}")
        
        # Try to get GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                header_lines.append(f"- **CUDA Available**: Yes")
                header_lines.append(f"- **CUDA Version**: {torch.version.cuda}")
                header_lines.append(f"- **GPU Count**: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    header_lines.append(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                header_lines.append(f"- **CUDA Available**: No")
        except Exception:
            header_lines.append(f"- **CUDA Available**: Unknown (torch not available)")
        
        # Try to get git info if available
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            if result.returncode == 0:
                header_lines.append(f"- **Git Commit**: {result.stdout.strip()}")
        except Exception:
            pass
        
        header_lines.append("")
        header_lines.append("---")
        header_lines.append("")
        
        # Write header
        header_file = os.path.join(self.report_dir, "header.md")
        with open(header_file, "w") as f:
            f.write("\n".join(header_lines))
        
        print(f"Report reset. Report directory: {self.report_dir}")
    
    def generate(self):
        """Generate the final report by combining all sections."""
        report_lines = []
        
        # Add header
        header_file = os.path.join(self.report_dir, "header.md")
        if os.path.exists(header_file):
            with open(header_file, "r") as f:
                report_lines.append(f.read())
        
        # Add sections in order
        section_files = sorted(Path(self.report_dir).glob("*.md"))
        for section_file in section_files:
            if section_file.name != "header.md" and section_file.name != "report.md":
                with open(section_file, "r") as f:
                    report_lines.append(f.read())
                report_lines.append("")
        
        # Add footer
        report_lines.append("---")
        report_lines.append("")
        report_lines.append(f"**Completed**: {datetime.now().isoformat()}")
        
        # Write final report
        report_content = "\n".join(report_lines)
        report_file = os.path.join(self.report_dir, "report.md")
        with open(report_file, "w") as f:
            f.write(report_content)
        
        # Also copy to current directory
        current_dir_report = os.path.join(os.getcwd(), "report.md")
        with open(current_dir_report, "w") as f:
            f.write(report_content)
        
        print(f"Report generated: {report_file}")
        print(f"Also copied to: {current_dir_report}")


# Global report instance
_report_instance: Report = None


def get_report() -> Report:
    """Get the global report instance."""
    global _report_instance
    if _report_instance is None:
        _report_instance = Report()
    return _report_instance


def main():
    """CLI entry point for report commands."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m nanochat.report [reset|generate]")
        sys.exit(1)
    
    command = sys.argv[1]
    report = get_report()
    
    if command == "reset":
        report.reset()
    elif command == "generate":
        report.generate()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python -m nanochat.report [reset|generate]")
        sys.exit(1)


if __name__ == "__main__":
    main()

