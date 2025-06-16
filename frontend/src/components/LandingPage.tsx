import { useNavigate } from 'react-router-dom';
import '../styles/LandingPage.css';

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      {/* Header Section */}
      <header className="header">
        <div className="logo">
          <h1>SmartSub</h1>
        </div>
        <nav className="nav">
          <ul>
            <li><a href="#features">Features</a></li>
            <li><a href="#testimonials">Testimonials</a></li>
            <li><a href="#sponsors">Sponsors</a></li>
            <li><a href="#contact">Contact</a></li>
          </ul>
        </nav>
      </header>

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <h1>Revolutionize Your Football Substitutions</h1>
          <p>SmartSub uses advanced AI to recommend optimal player substitutions in real-time, giving your team the competitive edge.</p>
          <button 
            className="cta-button"
            onClick={() => navigate('/home')}
          >
            Get Started
          </button>
        </div>
        <div className="hero-image">
          <img src="/images/hero-image.svg" alt="Football field with data visualization" />
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features">
        <h2>Why Choose SmartSub?</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Data-Driven Decisions</h3>
            <p>Leverage real-time analytics to make optimal substitution decisions.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3>Real-Time Analysis</h3>
            <p>Get instant recommendations based on live match data.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🏆</div>
            <h3>Proven Results</h3>
            <p>Teams using SmartSub have seen a 27% improvement in match outcomes.</p>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="testimonials" className="testimonials">
        <h2>What Coaches Say</h2>
        <div className="testimonials-container">
          <div className="testimonial">
            <p>"SmartSub transformed how we manage player rotations. It's like having an extra analyst on the bench."</p>
            <div className="testimonial-author">
              <strong>Carlos Riviera</strong>
              <span>Head Coach, FC Barcelona</span>
            </div>
          </div>
          <div className="testimonial">
            <p>"The data insights from SmartSub have given us a competitive edge in close matches."</p>
            <div className="testimonial-author">
              <strong>Emma Johnson</strong>
              <span>Technical Director, Arsenal WFC</span>
            </div>
          </div>
        </div>
      </section>

      {/* Sponsors Section */}
      <section id="sponsors" className="sponsors">
        <h2>Trusted By Industry Leaders</h2>
        <div className="sponsors-grid">
          <div className="sponsor">
            <img src="/images/nike.png" alt="Nike" />
          </div>
          <div className="sponsor">
            <img src="/images/adidas.png" alt="Adidas" />
          </div>
          <div className="sponsor">
            <img src="/images/puma.png" alt="Puma" />
          </div>
          <div className="sponsor">
            <img src="/images/under-armour.png" alt="Under Armour" />
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="cta">
        <div className="cta-content">
          <h2>Ready to Transform Your Team's Performance?</h2>
          <p>Join hundreds of professional teams already using SmartSub.</p>
          <button 
            className="cta-button"
            onClick={() => navigate('/home')}
          >
            Get Started Now
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer id="contact" className="footer">
        <div className="footer-content">
          <div className="footer-section">
            <h3>SmartSub</h3>
            <p>The ultimate football substitution recommendation system powered by AI.</p>
          </div>
          <div className="footer-section">
            <h3>Contact</h3>
            <p>Email: info@smartsub.com</p>
            <p>Phone: +1 (555) 123-4567</p>
          </div>
          <div className="footer-section">
            <h3>Follow Us</h3>
            <div className="social-links">
              <a href="#" className="social-link">Twitter</a>
              <a href="#" className="social-link">Facebook</a>
              <a href="#" className="social-link">Instagram</a>
            </div>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2023 SmartSub. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage; 